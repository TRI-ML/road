import numpy as np
import torch
from sklearn.neighbors import KDTree


def compute_knn(pcd_one, pcd_two):
    """
    Compute 3D loss between two point clouds computed as a mean of
    point pair distances between 2 point clouds.

    Args:
        pcd_one: First point cloud (N,3)
        pcd_two: Second point cloud (M,3)

    Returns: Distances (N) and indices (N) of nearest neighbors

    """
    if (pcd_one.size != 0) and (pcd_two.size != 0):
        # Estimate nearest neighbors (distances and ids) between given point clouds
        kdtree = KDTree(pcd_two)
        dists, idxs = kdtree.query(pcd_one)

        # Reformat distances and idxs to 1-dim array
        idxs = np.asarray([val for sublist in idxs for val in sublist])
        dists = np.asarray([val for sublist in dists for val in sublist])

    return dists, idxs


def normalize(V, F):
    """
    Normalize mesh to fit in a unit sphere.
    Args:
        V: Mesh vertices (N,3)
        F: Mesh faces (M,3)

    Returns: Scaled mesh vertices and faces

    """
    # Normalize mesh
    V_max, _ = torch.max(V, dim=0)
    V_min, _ = torch.min(V, dim=0)
    V_center = (V_max + V_min) / 2.
    V = V - V_center

    # Find the max distance to origin
    max_dist = torch.sqrt(torch.max(torch.sum(V ** 2, dim=-1)))
    V_scale = 1. / max_dist
    V *= V_scale
    return V, F


def subdivide(centers, level):
    """ Subdivide voxels given LoD

    Args:
        centers: Voxel centers (N,3)
        level: Level of detail

    Returns:
        centers_new: New voxel centers (N*8,3)
    """
    offset_size = (1 / (2 ** level)) * 2 / 4

    if isinstance(centers, np.ndarray):
        offsets = np.array([(-1, -1, -1), (-1, -1, 1), (-1, +1, -1), (-1, 1, 1), (1, -1, -1),
                            (1, -1, 1), (1, 1, -1), (1, 1, 1)]).astype(np.float32) * offset_size
        centers_new = np.repeat(centers, 8, axis=-2) + np.tile(offsets, (centers.shape[0], 1))
    elif torch.is_tensor(centers):
        offsets = torch.tensor([(-1, -1, -1), (-1, -1, 1), (-1, +1, -1), (-1, 1, 1), (1, -1, -1),
                                (1, -1, 1), (1, 1, -1), (1, 1, 1)]).to(centers.device) * offset_size
        centers_new = centers.repeat_interleave(8, dim=-2) + \
                      offsets.unsqueeze(0).repeat(centers.shape[0], centers.shape[1], 1)
    else:
        centers_new = None

    return centers_new


def get_cell_size(level):
    """ Get an octree cell size at given LoD """
    return (1 / (2 ** level)) * 2


def collate_fn(batch):
    """ Collate function for dataloader """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def to_cuda(gt, device):
    """ Move ground truth to GPU """
    # Loop over LoDs
    for l in [l for l in gt.keys() if isinstance(gt[l], dict)]:
        for k in gt[l].keys():
            gt[l][k] = gt[l][k].to(device)
    # Loop over other tensors
    for k in [k for k in gt.keys() if torch.is_tensor(gt[k])]:
        gt[k] = gt[k].to(device)
    return gt
