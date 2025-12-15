# Reference: diffusion is borrowed from the LDM repo: https://github.com/CompVis/latent-diffusion
# Specifically, functions from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py

import os
from collections import OrderedDict
from functools import partial

import numpy as np
from omegaconf import OmegaConf
from termcolor import colored
import torch
from torch import optim
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.diffusion_networks.network import DiffusionUNet
from models.model_utils import load_vqvae

from utils.util_3d import init_mesh_renderer, render_sdf, init_points_renderer, render_pcd

# ldm util
from models.networks.diffusion_networks.ldm_diffusion_util import (
    make_beta_schedule,
    extract_into_tensor,
    exists,
    default,
)
from models.networks.diffusion_networks.samplers.ddim import DDIMSampler

from transformers import BertConfig

import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEncoder
from models.emb_fun import get_embedder
import torch.nn.init as init
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(1000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # shape (1, max_seq_length, d_model)

    def forward(self, x):
        return self.pe[:, x].squeeze(0)

class BertEmbeddings_sincos(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        super().__init__()
        self.position_embeddings = PositionalEncoding(config.hidden_size, 300)  # relative point position from 0 to 300
        self.token_type_embeddings = PositionalEncoding(config.hidden_size, config.type_vocab_size) # stroke order from 0 to 300

        self.position_linear = nn.Linear(config.hidden_size, config.hidden_size) # project to hidden size
        self.token_type_linear = nn.Linear(config.hidden_size, config.hidden_size) # project to hidden size

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized

    def forward(
        self,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        embeddings = inputs_embeds

        position_embeddings = self.position_embeddings(position_ids)
        position_embeddings = self.position_linear(position_embeddings)
        embeddings += position_embeddings
        stroke_embeddings = self.token_type_embeddings(token_type_ids)
        stroke_embeddings = self.token_type_linear(stroke_embeddings)
        embeddings += stroke_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertTokenEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=768, seq_len=300, nerf_L=8, masking=True, ordering=True, num_layer=2):
        super().__init__()
        self.masking = masking
        self.ordering = ordering

        self.res, self.dim = get_embedder(nerf_L)

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        config = BertConfig(
            type_vocab_size=500,
            hidden_size=hidden_dim,
            num_attention_heads=8,
            num_hidden_layers=num_layer,
            intermediate_size=hidden_dim,
            max_position_embeddings=seq_len + 10,
            attn_implementation="eager"
        )
        self.embedding_layer = BertEmbeddings_sincos(config)
        self.encoder = BertEncoder(config)

        self.end_token_index = nn.Parameter(torch.empty(1, hidden_dim))
        init.xavier_uniform_(self.end_token_index)

        self.sep_token_index = nn.Parameter(torch.empty(1, hidden_dim))
        init.xavier_uniform_(self.sep_token_index)

        if self.masking:
            self.mask_token_index = nn.Parameter(torch.empty(1, hidden_dim))
            init.xavier_uniform_(self.mask_token_index)

    def get_extended_attention_mask(self, attention_mask):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask: {attention_mask.shape}")

        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, input_dict):
        x, attention_mask, token_ids, end_token_index, sep_token_index, mask_token_index, position_ids = input_dict['input_ids'], input_dict['attention_mask'], input_dict['token_type_ids'], input_dict['end_token_index'], input_dict['sep_token_index'], input_dict['mask_token_index'], input_dict['position_ids']
        pose_embed_loc = self.res(x) # [B, N, 12]
        input_embeddings = self.proj(pose_embed_loc)  # [B, N, 768]

        end_mask = end_token_index != 0
        sep_mask = sep_token_index != 0

        input_embeddings[end_mask] = self.end_token_index
        input_embeddings[sep_mask] = self.sep_token_index

        if self.masking:
            mask_mask = mask_token_index != 0
            input_embeddings[mask_mask] = self.mask_token_index

        if self.ordering:
            x = self.embedding_layer(inputs_embeds=input_embeddings, position_ids=position_ids, token_type_ids=token_ids)
        else:
            x = input_embeddings

        extended_mask = self.get_extended_attention_mask(attention_mask)
        out = self.encoder(x, attention_mask=extended_mask)
        return out


class SDFusionSketch2ShapeModel(BaseModel):
    def name(self):
        return 'SDFusion-Sketch2Shape-Model'

    def initialize(self, lr, seq_len, nerf_L, bert_input_dim, bert_hidden_dim, masking, ordering, num_layers, ckpt=None):
        BaseModel.initialize(self)
        self.lr = lr
        self.isTrain = True

        self.model_name = self.name()
        self.device = 'cuda'

        self.df_cfg = '../configs/sdfusion-sketch2shape.yaml'
        self.vq_cfg='../configs/vqvae_snet.yaml'

        ######## START: Define Networks ########
        assert self.df_cfg is not None
        assert self.vq_cfg is not None

        # init df
        df_conf = OmegaConf.load(self.df_cfg)
        vq_conf = OmegaConf.load(self.vq_cfg)

        # record z_shape
        ddconfig = vq_conf.model.params.ddconfig
        shape_res = ddconfig.resolution
        z_ch, n_down = ddconfig.z_channels, len(ddconfig.ch_mult)-1
        z_sp_dim = shape_res // (2 ** n_down)
        self.z_shape = (z_ch, z_sp_dim, z_sp_dim, z_sp_dim)

        df_model_params = df_conf.model.params
        unet_params = df_conf.unet.params
        self.df = DiffusionUNet(unet_params, vq_conf=vq_conf, conditioning_key=df_model_params.conditioning_key)
        self.df.to(self.device)
        self.init_diffusion_params(uc_scale=3.)
        
        # sampler
        self.ddim_sampler = DDIMSampler(self)
        
        # init vqvae
        self.vq_ckpt="../saved_ckpt/vqvae-snet-all.pth"

        self.vqvae = load_vqvae(vq_conf, vq_ckpt=self.vq_ckpt, device=self.device)

        model = BertTokenEncoder(input_dim=bert_input_dim, hidden_dim=bert_hidden_dim, seq_len=seq_len, nerf_L=nerf_L, masking=masking, ordering=ordering, num_layer=num_layers)
        self.cond_model = model

        self.cond_model.to(self.device)
        for param in self.cond_model.parameters():
            param.requires_grad = True

        ######## END: Define Networks ########
        # param list
        trainable_models = [self.df, self.cond_model]
        trainable_params = []
        for m in trainable_models:
            trainable_params += [p for p in m.parameters() if p.requires_grad == True]

        if self.isTrain:
            # initialize optimizers
            self.optimizer = optim.AdamW(trainable_params, lr=self.lr)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        self.ckpt = ckpt
        if self.ckpt is not None:
            self.load_ckpt(self.ckpt, load_opt=self.isTrain)

        # transforms
        self.to_tensor = transforms.ToTensor()

        dist, elev, azim = 2.5, 40, 20+270 # shapenet
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.device)
        self.renderer_pcd = init_points_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.device)

        self.df_module = self.df
        self.vqvae_module = self.vqvae
        self.cond_model_module = self.cond_model

        # for debugging purpose
        self.ddim_steps = 100

    ############################ START: init diffusion params ############################
    def init_diffusion_params(self, uc_scale=3.):
        df_conf = OmegaConf.load(self.df_cfg)
        df_model_params = df_conf.model.params
        
        # ref: ddpm.py, line 44 in __init__()
        self.parameterization = "eps"
        self.learn_logvar = False
        
        self.v_posterior = 0.
        self.original_elbo_weight = 0.
        self.l_simple_weight = 1.
        # ref: ddpm.py, register_schedule
        self.register_schedule(
            timesteps=df_model_params.timesteps,
            linear_start=df_model_params.linear_start,
            linear_end=df_model_params.linear_end,
        )
        
        logvar_init = 0.
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,)).to(self.device)
        # for cls-free guidance
        self.uc_scale = uc_scale

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                        linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.betas = to_torch(betas).to(self.device)
        self.alphas_cumprod = to_torch(alphas_cumprod).to(self.device)
        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev).to(self.device)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod)).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod)).to(self.device)
        self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod)).to(self.device)
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod)).to(self.device)
        self.sqrt_recipm1_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod - 1)).to(self.device)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = to_torch(posterior_variance).to(self.device)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = to_torch(np.log(np.maximum(posterior_variance, 1e-20))).to(self.device)
        self.posterior_mean_coef1 = to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to(self.device)
        self.posterior_mean_coef2 = to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)).to(self.device)

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas).to(self.device) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = lvlb_weights
        assert not torch.isnan(self.lvlb_weights).all()
        ############################ END: init diffusion params ############################

    def set_input(self, input=None, max_sample=None, mode='train'):
        if mode == 'val' or mode == 'test':
            self.x = input['sdf'].to(self.device)
            vars_list = ['x']
            self.tocuda(var_names=vars_list)

        self.z = input['latent_z'].to(self.device)
        self.sketch = {key: tensor.to(self.device) for key, tensor in input['sketch'].items()}
        self.uc_sketch = {key: torch.zeros_like(tensor).to(self.device) for key, tensor in self.sketch.items()}

        if max_sample is not None:
            self.x = self.x[:max_sample]
            self.sketch = self.sketch[:max_sample]
            self.uc_sketch = self.uc_sketch[:max_sample]

    def switch_train(self):
        self.df.train()
        self.cond_model.train()

    def switch_eval(self):
        self.df.eval()
        self.vqvae.eval()
        self.cond_model.eval()

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    # check: ddpm.py, line 891
    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.df_module.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        # eps
        out = self.df(x_noisy, t, **cond)

        if isinstance(out, tuple) and not return_ids:
            return out[0]
        else:
            return out

    def get_loss(self, pred, target, loss_type='l2', mean=True):
        if loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    # check: ddpm.py, line 871 forward
    # check: p_losses
    # check: q_sample, apply_model
    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # predict noise (eps) or x0

        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        # l2
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        loss_dict.update({f'loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'loss_total': loss.clone().detach().mean()})

        return x_noisy, target, loss, loss_dict


    def forward(self):
        self.switch_train()

        outputs = self.cond_model(self.sketch) # B, 15000, 1088
        c_sketch = outputs.last_hidden_state  # shape: (batch_size, hidden_dim)

        z = self.z
        # 2. do diffusion's forward
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()

        z_noisy, target, loss, loss_dict = self.p_losses(z, c_sketch, t)

        self.loss_df = loss
        self.loss_dict = loss_dict

    # check: ddpm.py, log_images(). line 1317~1327
    @torch.no_grad()
    def inference(self, data, ddim_steps=None, ddim_eta=0., uc_scale=None,
                  infer_all=False, max_sample=16):
        self.switch_eval()

        if not infer_all:
            self.set_input(data, max_sample=max_sample)
        else:
            self.set_input(data)

        if ddim_steps is None:
            ddim_steps = self.ddim_steps

        if uc_scale is None:
            uc_scale = self.uc_scale

        # get noise, denoise, and decode with vqvae
        outputs = self.cond_model(self.sketch)
        c_sketch = outputs.last_hidden_state

        B = c_sketch.shape[0]
        shape = self.z_shape
        samples, intermediates = self.ddim_sampler.sample(S=ddim_steps,
                                                     batch_size=B,
                                                     shape=shape,
                                                     conditioning=c_sketch,
                                                     verbose=False,
                                                     eta=ddim_eta)
        
        # decode z
        self.gen_df = self.vqvae_module.decode_no_quant(samples)
        self.switch_train()

    @torch.no_grad()
    def sketch2shape(self, input_sketch, ngen=6, ddim_steps=100, ddim_eta=0.0, uc_scale=None):

        self.switch_eval()

        data = {
            'sdf': torch.zeros(ngen),
            'sketch': [input_sketch] * ngen,
        }
        
        self.set_input(data)

        ddim_sampler = DDIMSampler(self)
        
        if ddim_steps is None:
            ddim_steps = self.ddim_steps

        if uc_scale is None:
            uc_scale = self.scale
            
        # get noise, denoise, and decode with vqvae
        uc = self.cond_model(self.uc_sketch)
        c_sketch = self.cond_model(self.sketch)
        B = c_sketch.shape[0]
        shape = self.z_shape
        samples, intermediates = ddim_sampler.sample(S=ddim_steps,
                                                     batch_size=B,
                                                     shape=shape,
                                                     conditioning=c_sketch,
                                                     verbose=False,
                                                     unconditional_guidance_scale=uc_scale,
                                                     unconditional_conditioning=uc,
                                                     eta=ddim_eta)
        
        # decode z
        self.gen_df = self.vqvae_module.decode_no_quant(samples)
        return self.gen_df

    def backward(self):
        self.loss = self.loss_df
        self.loss_total = self.loss_dict['loss_total']
        self.loss_simple = self.loss_dict['loss_simple']
        self.loss_vlb = self.loss_dict['loss_vlb']
        if 'loss_gamma' in self.loss_dict:
            self.loss_gamma = self.loss_dict['loss_gamma']

        self.loss.backward()

    def optimize_parameters(self):
        self.set_requires_grad([self.df], requires_grad=True)
        self.set_requires_grad([self.cond_model], requires_grad=True)

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_current_errors(self):
        ret = OrderedDict([
            ('total', self.loss_total.mean().data),
            ('simple', self.loss_simple.mean().data),
            ('vlb', self.loss_vlb.mean().data),
        ])

        if hasattr(self, 'loss_gamma'):
            ret['gamma'] = self.loss_gamma.mean().data

        return ret

    def get_current_obj(self):
        with torch.no_grad():
            _, self.obj_gt = render_sdf(self.renderer, self.x)
            _, self.obj_gen_df = render_sdf(self.renderer, self.gen_df, level=0.005)

        return self.obj_gt, self.obj_gen_df

    def get_current_visuals(self):
        with torch.no_grad():
            self.sketch = self.sketch # input sketch (8, 3, 2500)
            self.img_gt, _ = render_sdf(self.renderer, self.x)
            self.img_gt = self.img_gt.detach().cpu()
            self.img_gen_df, _ = render_sdf(self.renderer, self.gen_df)
            self.img_gen_df = self.img_gen_df.detach().cpu()

            self.img_pcd = render_pcd(self.renderer_pcd, self.sketch['input_ids'], color=[1, 1, 1], alpha=False)

        vis_tensor_names = [
            'img_gt',
            'img_gen_df',
            'img_pcd'
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)

        return OrderedDict(visuals)

    def save(self, label, global_step, ckpt_dir, save_opt=False):
        state_dict = {
            'vqvae': self.vqvae_module.state_dict(),
            'cond_model': self.cond_model_module.state_dict(),
            'df': self.df_module.state_dict(),
            'global_step': global_step,
        }
        
        if save_opt:
            state_dict['opt'] = self.optimizer.state_dict()

        save_filename = 'df_%s.pth' % (label)
        save_path = os.path.join(ckpt_dir, save_filename)

        torch.save(state_dict, save_path)

    @staticmethod
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for key in state_dict.keys():  # key 比如 'vqvae', 'sketch_encoder', ...
            if key == 'global_step':
                continue
            sub_dict = state_dict[key]
            new_sub_dict = {}
            for k, v in sub_dict.items():
                if k.startswith('module.'):
                    new_sub_dict[k[7:]] = v  # 去掉 'module.'
                else:
                    new_sub_dict[k] = v
            new_state_dict[key] = new_sub_dict
        return new_state_dict

    def load_ckpt(self, ckpt, load_opt=False):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn, weights_only=True)
        else:
            state_dict = ckpt

        state_dict = self.remove_module_prefix(state_dict)

        # positional embeddings are not learned, so skip loading them
        keys_to_delete = ['embedding_layer.position_embeddings.pe', 'embedding_layer.token_type_embeddings.pe']
        for k in keys_to_delete:
            if k in state_dict['cond_model']:
                print(f"⚠️ Skip loading: {k}")
                del state_dict['cond_model'][k]

        self.vqvae.load_state_dict(state_dict['vqvae'])
        self.df.load_state_dict(state_dict['df'])
        self.cond_model.load_state_dict(state_dict['cond_model'], strict=False)
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

        if load_opt:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))

