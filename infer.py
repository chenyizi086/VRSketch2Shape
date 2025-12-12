import argparse
import os
import torch

import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader
from termcolor import cprint
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.io import save_obj

from models.sketch2shape_model import SDFusionSketch2ShapeModel
from utils.visualizer import Visualizer
from eval.eval_obj import calculate_fscore_pytorch3d, normalize_to_box
from dataloader.sketch_data import Sketch2ShapeDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Training Configurations")

    # These parameters are fixed
    parser.add_argument("--seed", type=int, default=10, help="Random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--seq_len", type=int, default=1200, help="Max sequence length")
    parser.add_argument("--mask_per", type=float, default=0.15, help="Mask percentage")
    parser.add_argument("--bert_hidden_dim", type=int, default=256, help="Hidden dim of BERT encoder")

    # These parameters will be changed in the experiments
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for training")
    parser.add_argument("--exp_name", type=str, default="sketch2shape_experiment", help="Experiment name for logging")
    parser.add_argument("--dataset", type=str, choices=["syn", "real"], default="syn", help="Dataset name: 'syn' for synthetic or 'real' for real sketches")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the pre-trained model checkpoint")

    parser.add_argument("--nerf_L", type=int, default=10, help="NeRF L frequency parameter")
    parser.add_argument("--bert_input_dim", type=int, default=51, help="Input dim of BERT encoder")
    parser.add_argument("--masking", action='store_true', help="Whether to use masking in BERT")
    parser.add_argument("--ordering", action='store_true', help="Whether to use ordering in BERT")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers in BERT encoder")

    parser.add_argument("--number_sketches", type=int, default=None, help="Number of sketches to use in the dataset (only for 'real' dataset)")

    return parser.parse_args()

def train():
    # Create the dataloader
    obj_idx = ['02691156', '02933112', '03001627', '04379243']
    data_dir = '/mnt/d/sdfusion/sketch3D_final'
    sdf_path = os.path.join(data_dir, 'data/sdf')
    sketch_path = os.path.join(data_dir, 'data')
    latent_z_path = os.path.join(data_dir, 'data/z_code_0.2')
    args = parse_args()

    # fix seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    lr = args.lr
    seq_len = args.seq_len
    mask_per = args.mask_per
    nerf_L = args.nerf_L
    bert_input_dim = args.bert_input_dim
    bert_hidden_dim = args.bert_hidden_dim
    masking = args.masking
    ordering = args.ordering
    num_layers = args.num_layers
    data_type = args.dataset  # 'syn' or 'real'

    # if not using masking, set mask_per to None
    if not(masking):
        print('not using masking')
        mask_per = None
    else:
        print('using masking with mask_per {}'.format(mask_per))
    
    if not(ordering):
        print('not using ordering')
    else:
        print('using ordering')

    # Create model
    model = SDFusionSketch2ShapeModel()
    model.initialize(lr, seq_len, nerf_L, bert_input_dim, bert_hidden_dim, masking, ordering, num_layers)
    cprint("[*] Model has been created: %s" % model.name(), 'blue')

    # visualizer
    model_name = args.exp_name
    visualizer = Visualizer(isTrain=False, name=model_name)
    visualizer.setup_io()

    # Load the model
    model_path = args.model_path
    model.load_ckpt(model_path)
    cprint("[*] Model has been loaded: %s" % model_path, 'blue')

    print('using real sketch data')

    tes_dl = Sketch2ShapeDataset(res=64, obj_idx=obj_idx, sdf_path=sdf_path, sketch_path=sketch_path, latent_z_path=latent_z_path, seq_len=seq_len, mask_per=mask_per, data_type=data_type)
    test_dataloader = DataLoader(tes_dl, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, drop_last=False)
    
    save_path = os.path.join(Path(visualizer.img_dir).parent, 'visualization_3D')
    if not os.path.exists(save_path):
        print('create folder: {}'.format(save_path))
        os.makedirs(save_path)

    num_samples = 4096
    chamfer_distance_list = []
    F_score_list = []
    for index, data in enumerate(test_dataloader):
        model.set_input(data, mode='test')
        model.inference(data, ddim_steps=100, infer_all=True)
        obj_gt, obj_gen_df = model.get_current_obj()

        for i, (verts, faces) in enumerate(zip(obj_gen_df.verts_list(), obj_gen_df.faces_list())):
            # create save directories
            cls_idx = data['name'][i].split('/')[-3]
            obj_idx = data['name'][i].split('/')[-2]
            if not os.path.exists(f"{save_path}/{cls_idx}/gen_df"):
                os.makedirs(f"{save_path}/{cls_idx}/gen_df", exist_ok=True)
            
            if not os.path.exists(f"{save_path}/{cls_idx}/gt"):
                os.makedirs(f"{save_path}/{cls_idx}/gt", exist_ok=True)

            save_obj(f"{save_path}/{cls_idx}/gen_df/{obj_idx}.obj", verts, faces)
            save_obj(f"{save_path}/{cls_idx}/gt/{obj_idx}.obj", obj_gt.verts_list()[i], obj_gt.faces_list()[i])

        pointcloud_gen = sample_points_from_meshes(obj_gen_df, num_samples)
        pointcloud_gt  = sample_points_from_meshes(obj_gt, num_samples)

        # Chamfer distance
        normal_pointcloud_gen = normalize_to_box(pointcloud_gen)[0]
        normal_pointcloud_gt = normalize_to_box(pointcloud_gt)[0]

        cd_dist = chamfer_distance(normal_pointcloud_gen, normal_pointcloud_gt, batch_reduction=None)[0]
        chamfer_distance_list.append(cd_dist.mean().item())
        print('Index: {}, Chamfer distance: {}'.format(index, cd_dist.mean().item()))

        fscore, precision, recall = calculate_fscore_pytorch3d(normal_pointcloud_gt, normal_pointcloud_gen, th=2*0.02)
        print('Index: {}, F-score: {}, Precision: {}, Recall: {}'.format(index, fscore, precision, recall))
        F_score_list.append(fscore)

        visualizer.display_current_results(model.get_current_visuals(), 0, index, phase='test')
    print('Average Chamfer distance: {}'.format(np.mean(chamfer_distance_list)))
    print('Average F-score: {}'.format(np.mean(F_score_list)))

if __name__ == '__main__':
    train()
