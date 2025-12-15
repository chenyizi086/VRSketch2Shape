import numpy as np
from simplifyline import simplify_line_3d, MatrixDouble
import torch.utils.data as data
import h5py
import glob
import json
import torch
import os


def generate_random_mask(length, mask_percent, mask_percent_point=0.3):
    # 15% strokes fully masked
    num_mask = max(1, int(length * mask_percent))
    mask_indices = np.random.choice(length, num_mask, replace=False)

    # exclude last token (SEP+EOS combined)
    remaining = np.setdiff1d(np.arange(length-1), mask_indices)

    # 30% of remaining points masked
    num_mask_pc = max(1, int(len(remaining) * mask_percent_point))
    mask_pointcloud_index = np.random.choice(remaining, num_mask_pc, replace=False)

    return mask_indices, mask_pointcloud_index


def generate_random_point(length, mask_percent_point=0.3):
    # 30% points are randomly choiced to be masked for points
    num_mask = max(1, int(length * mask_percent_point))
    # skip the last token becuase it is always [EOS]
    mask_indices_point = np.random.choice(length-1, num_mask, replace=False)
    mask_list = np.zeros(length, dtype=int)
    mask_list[mask_indices_point] = 1
    mask_list = mask_list.tolist()
    return mask_list


def sketch_line_to_token_high_dim(lines, seq_len, mask_percent=0.15):
    token = []
    token_ids = []
    end_token_index = []
    sep_token_index = []
    mask_token_index = []
    position_ids = []

    # Generate random stroke-level and point-level masks
    if mask_percent:
        random_stroke_mask, random_point_mask = generate_random_mask(len(lines), mask_percent)

    token_count = 1  # Stroke index (token_type_id)

    for i, line_seg in enumerate(lines):
        line_token = []
        pos_count = 1  # Position index for positional encoding

        for j, point in enumerate(line_seg):

            # Handle last stroke and last point: add [SEP] and [EOS]
            if i == len(lines) - 1 and j == len(line_seg) - 1:
                # Real point
                line_token.append(point)
                sep_token_index.append(0)
                end_token_index.append(0)
                token_ids.append(token_count)
                position_ids.append(pos_count)
                pos_count += 1

                # [SEP]
                line_token.append([0, 0, 0])
                sep_token_index.append(1)
                end_token_index.append(0)
                token_ids.append(token_count)
                position_ids.append(pos_count)
                pos_count += 1

                # [EOS]
                line_token.append([0, 0, 0])
                sep_token_index.append(0)
                end_token_index.append(1)
                token_ids.append(token_count)
                position_ids.append(pos_count)
                pos_count += 1

            # End of a normal stroke: add [SEP]
            elif j == len(line_seg) - 1:
                line_token.append(point)
                sep_token_index.append(0)
                end_token_index.append(0)
                token_ids.append(token_count)
                position_ids.append(pos_count)
                pos_count += 1

                line_token.append([0, 0, 0])
                sep_token_index.append(1)
                end_token_index.append(0)
                token_ids.append(token_count)
                position_ids.append(pos_count)
                pos_count += 1

            # Normal point
            else:
                line_token.append(point)
                sep_token_index.append(0)
                end_token_index.append(0)
                token_ids.append(token_count)
                position_ids.append(pos_count)
                pos_count += 1

        # Handle masking
        if mask_percent:
            if i in random_stroke_mask:
                mask_token_index += [1] * len(line_token)
            elif i in random_point_mask:
                if len(line_token) <= 5:
                    mask_token_index += [0] * len(line_token)
                else:
                    mask_token_index += generate_random_point(len(line_token))
            else:
                mask_token_index += [0] * len(line_token)
        else:
            mask_token_index += [0] * len(line_token)

        token.append(np.array(line_token))
        token_count += 1

    # Concatenate strokes
    token = np.concatenate(token)

    # Apply mask: zero out masked points
    token = token * (1 - np.array(mask_token_index)[:, None])

    max_length = seq_len
    # Padding token
    real_token_length = len(token)
    if real_token_length < max_length:
        # print('padding from {} to {}'.format(real_token_length, max_length))
        padding_length = max_length - real_token_length
        padding = np.zeros((padding_length, token.shape[1]), dtype=np.float32)
        token = np.concatenate((token, padding), axis=0)
        
        attention_mask = [1] * real_token_length + [0] * padding_length
        attention_mask = np.array(attention_mask, dtype=np.float32)

        token_ids = np.concatenate((token_ids, np.zeros(padding_length, dtype=np.int64)), axis=0)
        
        end_token_index = np.concatenate((end_token_index, np.zeros(padding_length, dtype=np.int64)), axis=0)
        sep_token_index = np.concatenate((sep_token_index, np.zeros(padding_length, dtype=np.int64)), axis=0)
        mask_token_index = np.concatenate((mask_token_index, np.zeros(padding_length, dtype=np.int64)), axis=0)
        position_ids = np.concatenate((position_ids, np.zeros(padding_length, dtype=np.int64)), axis=0)
    else:
        print('precision issue! token length {} exceeds max length {}'.format(real_token_length, max_length))
        token = token[:max_length]
        attention_mask = [1] * max_length
        attention_mask = np.array(attention_mask, dtype=np.float32)
        token_ids = token_ids[:max_length]
        end_token_index = end_token_index[:max_length]
        sep_token_index = sep_token_index[:max_length]
        mask_token_index = mask_token_index[:max_length]
        position_ids = position_ids[:max_length]

    token = torch.tensor(token, dtype=torch.float32)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float32)
    token_type_ids = torch.tensor(token_ids, dtype=torch.long)  # All tokens are of type 0
    end_token_index = torch.tensor(end_token_index, dtype=torch.long)
    sep_token_index = torch.tensor(sep_token_index, dtype=torch.long)
    mask_token_index = torch.tensor(mask_token_index, dtype=torch.long)
    position_ids = torch.tensor(position_ids, dtype=torch.long)

    output = {
        'input_ids': token,  # Add batch dimension
        'attention_mask': attention_mask,  # Add batch dimension
        'token_type_ids': token_type_ids,  # Add batch dimension
        'end_token_index': end_token_index,
        'sep_token_index': sep_token_index,
        'mask_token_index': mask_token_index,
        'position_ids': position_ids
    }
    return output

def normalize_lines_to_unit_sphere(lines):
    # Flatten all points into one array
    all_points = np.concatenate(lines, axis=0)  # shape (total_points, 3)

    center = np.mean(all_points, axis=0)
    all_points -= center

    max_dist = np.max(np.linalg.norm(all_points, axis=1))
    all_points /= max_dist

    normalized_lines = []
    idx = 0
    for line in lines:
        n = len(line)
        normalized_lines.append(all_points[idx:idx + n].tolist())
        idx += n

    return normalized_lines

def simplify_lines(lines, simplification_threshold):
    s_lines = []
    for l in lines:
        l_mat = MatrixDouble(l)
        simplified_mat = simplify_line_3d(l_mat, max_distance=simplification_threshold, high_quality=True)
        simplified = np.array(simplified_mat)
        s_lines.append(simplified)
    return s_lines


class Sketch2ShapeDataset(data.Dataset):
    def __init__(self, res, obj_idx, sdf_path, sketch_path, latent_z_path, seq_len, mask_per=0.15, data_type='real'):
        self.sketch_path = sketch_path
        self.seq_len = seq_len
        self.latent_z_path = latent_z_path
        self.data_type = data_type
        self.mask_per = mask_per
        self.sdf_path = sdf_path
        self.sketch_list = []
        self.pc_list = []
        self.sdf_list = []

        for idx in obj_idx:
            sketch_dir = os.path.join(sketch_path, idx)
            latent_dir = os.path.join(latent_z_path, idx)
            for i in os.listdir(os.path.join(sketch_dir, 'test')):
                try:
                    strokes = glob.glob(os.path.join(sketch_dir, 'test', i, 'Detail_*', 'Strokes.curves'))[0]
                except IndexError:
                    strokes = None

                latent_z = os.path.join(latent_dir, i + '.npz')
                h5file = os.path.join(self.sdf_path, idx, i, 'ori_sample_grid.h5')

                if os.path.exists(strokes) and os.path.exists(latent_z) and os.path.exists(h5file):
                    self.sketch_list.append(strokes)
                    self.pc_list.append(latent_z)
                    self.sdf_list.append(h5file)

            print('found {} real sketch samples'.format(len(self.sketch_list)))

        self.trunc_thres = 0.2
        self.res = res

    def read_syn_lines(self, lerp_strokes):
        with open(lerp_strokes, 'r') as f:
            ori_lines = json.load(f)
        
        lines = [np.array(line) for line in ori_lines]
        return lines

    def read_real_lines(self, lerp_strokes):
        lines = []
        line_seg = []
        with open(lerp_strokes, 'r') as file:
            for line in file:
                if line.startswith('v'):
                    lines.append(np.array(line_seg))
                    line_seg = []
                else:
                    x, z, y = line.split(' ')
                    x, y, z = -float(x), float(y), float(z)
                    line_seg.append([x, y, z])
        lines.append(np.array(line_seg))  # Append the last segment
        lines = lines[1:]  # Remove first empty line
        return lines

    def __getitem__(self, index):
        z = np.load(self.pc_list[index])['z']
        z = torch.tensor(z, dtype=torch.float32)

        lerp_strokes = self.sketch_list[index]
        if self.data_type == 'real':
            lines = self.read_real_lines(lerp_strokes)
            lines_simply = simplify_lines(lines, 0.5)
        elif self.data_type == 'syn':
            lines = self.read_syn_lines(lerp_strokes)
            lines_simply = simplify_lines(lines, 0.01)
        if len(lines_simply) == 0:
            print(self.sketch_list[index])
            raise ValueError("No lines provided to normalize.")
        normalized_lines = normalize_lines_to_unit_sphere(lines_simply)
        sketch_token = sketch_line_to_token_high_dim(normalized_lines, self.seq_len, mask_percent=self.mask_per)

        h5_f = h5py.File(self.sdf_list[index], 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, self.res, self.res, self.res)

        thres = self.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)
        ret = {
            'sketch': sketch_token,
            'latent_z': z,
            'sdf': sdf,
            'name': self.sketch_list[index]
        }
        return ret
    
    def __len__(self):
        return len(self.sketch_list)

    def name(self):
        return 'Sketch2ShapeDataset'