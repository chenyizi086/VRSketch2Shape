
import numpy as np
import torch
from pytorch3d.ops import knn_points


# ---------------------- PyTorch3D Version ----------------------
def calculate_fscore_pytorch3d(gt_points: torch.Tensor, pr_points: torch.Tensor, th: float = 0.01):
    if gt_points.dim() == 2:
        gt_points = gt_points.unsqueeze(0)
    if pr_points.dim() == 2:
        pr_points = pr_points.unsqueeze(0)

    d1 = knn_points(gt_points, pr_points, K=1).dists.sqrt().squeeze(-1)
    d2 = knn_points(pr_points, gt_points, K=1).dists.sqrt().squeeze(-1)

    recall = (d2 < th).float().mean(dim=1)
    precision = (d1 < th).float().mean(dim=1)
    fscore = 2 * precision * recall / (precision + recall + 1e-8)

    recall = recall.mean()
    precision = precision.mean()
    fscore = fscore.mean()

    return fscore.item(), precision.item(), recall.item()


def normalize_to_box(input):
    """
    normalize point cloud to unit bounding box
    center = (max - min)/2
    scale = max(abs(x))
    input: pc [N, P, dim] or [P, dim]
    output: pc, centroid, furthest_distance

    From https://github.com/yifita/pytorch_points
    """
    if len(input.shape) == 2:
        axis = 0
        P = input.shape[0]
        D = input.shape[1]
    elif len(input.shape) == 3:
        axis = 1
        P = input.shape[1]
        D = input.shape[2]
    else:
        raise ValueError()
    
    if isinstance(input, np.ndarray):
        maxP = np.amax(input, axis=axis, keepdims=True)
        minP = np.amin(input, axis=axis, keepdims=True)
        centroid = (maxP+minP)/2
        input = input - centroid
        furthest_distance = np.amax(np.abs(input), axis=(axis, -1), keepdims=True)
        input = input / furthest_distance
    elif isinstance(input, torch.Tensor):
        maxP = torch.max(input, dim=axis, keepdim=True)[0]
        minP = torch.min(input, dim=axis, keepdim=True)[0]
        centroid = (maxP+minP)/2
        input = input - centroid
        in_shape = list(input.shape[:axis])+[P*D]
        furthest_distance = torch.max(torch.abs(input).reshape(in_shape), dim=axis, keepdim=True)[0]
        furthest_distance = furthest_distance.unsqueeze(-1)
        input = input / furthest_distance
    else:
        raise ValueError()

    return input, centroid, furthest_distance