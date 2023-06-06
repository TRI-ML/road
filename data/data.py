import os

import numpy as np
import open3d as o3d
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from data.utils import compute_knn, normalize, subdivide, get_cell_size


class OctDB(Dataset):
    """ Main dataset class """

    def __init__(self, args, split, num_samples=1000000):
        self.path = args.path_data
        self.offline = False

        # Create octgrid
        self.lods = args.lods
        self.lod_current = args.lods
        self.test = split == 'testing'

        supported_formats = ['.ply', '.obj', '.stl', '.off']
        path_octdb = os.path.join(self.path, 'octdb')
        os.makedirs(path_octdb, exist_ok=True)

        self.models = []
        for root, directories, filenames in os.walk(self.path):
            pbar = tqdm(filenames)
            for file in pbar:
                if os.path.splitext(file)[1] in supported_formats:
                    # Load cached data or generate new
                    file_npy = os.path.join(path_octdb, file[:-3] + 'npy')
                    if os.path.isfile(file_npy):
                        model = np.load(file_npy, allow_pickle=True).item()
                    else:
                        # load model
                        model_o3d_mesh = o3d.io.read_triangle_mesh(os.path.join(self.path, file))
                        V, F = normalize(torch.from_numpy(np.array(model_o3d_mesh.vertices)),
                                         torch.from_numpy(np.array(model_o3d_mesh.triangles)).long())
                        model_o3d_mesh.vertices = o3d.utility.Vector3dVector(V.numpy())
                        model_o3d_mesh.compute_vertex_normals()
                        model_o3d = model_o3d_mesh.sample_points_uniformly(number_of_points=num_samples)
                        model_o3d.normalize_normals()
                        model = self._build_feature_octree(np.array(model_o3d.points), np.array(model_o3d.normals),
                                                           file[4:-4], lods=args.lods)

                        # For metrics
                        model['pcd'] = np.array(model_o3d.points).astype(np.float32)
                        model['nrm'] = np.array(model_o3d.normals).astype(np.float32)

                        # Save file
                        np.save(file_npy, model)

                    # Store model
                    self.models.append(model)

    def _build_feature_octree(self, points, normals, idx, lods=5):
        """ Build a ground truth octree of a predefined level of detail (LoD) """
        centers = np.array([[0., 0., 0.]])
        model = {}
        model['pcd'] = points
        model['nrm'] = normals
        model['idx'] = idx

        for lod in range(lods + 1):
            dist, idxs = compute_knn(centers, model['pcd'])
            # Prune voxels that are more than the cell size away from the surface
            centers_occ = dist <= get_cell_size(lod)
            centers_filtered = centers[centers_occ]

            # Correct SDF for the surface
            surface2grid = centers - model['pcd'][idxs]
            dist = (normals[idxs] * surface2grid).sum(-1)  # dot product

            # Add annotations per level
            model[lod] = {}
            model[lod]['xyz'] = centers.astype(np.float32)
            model[lod]['occ'] = centers_occ
            model[lod]['sdf'] = dist.astype(np.float32)
            model[lod]['nrm'] = normals[idxs].astype(np.float32)

            # Get to the next level
            centers = subdivide(centers_filtered, lod)

        return model

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):

        if self.offline:
            model = np.load(self.models[idx]['path'], allow_pickle=True).item()
        else:
            model = self.models[idx]

        # Generate batch data
        output = {}
        output['idx'] = idx
        output['name'] = model['idx']
        output[0] = {}
        output[0]['ids'] = int(np.zeros([1]))
        output[0]['occ'] = np.ones([1]).astype(bool)
        output[0]['xyz'] = np.zeros([1, 3])

        lod_ids_local_all = []
        lod_ids_local_all.append(np.arange(model[1]['xyz'].shape[0]))

        if self.test:
            output['pcd'] = model['pcd']
            output['nrm'] = model['nrm']
        else:

            # Randomly sample N points from each level
            for lod in range(1, self.lod_current + 1):
                n_points = min(((2 ** lod) ** 3), 2 ** 10)
                lod_ids_global = np.arange(model[lod]['xyz'].shape[0])  # LoD point ids
                lod_ids_global_filt = lod_ids_global[model[lod]['occ']]  # LoD ids after occ filtering
                lod_ids_local = lod_ids_local_all[-1]  # LoD local point ids (after random selection from prev. level)
                lod_occ_local = model[lod]['occ'][lod_ids_local]  # LoD occlusions
                if len(np.arange(lod_ids_local.shape[0])[lod_occ_local]) > 0:
                    lod_ids_local_rnd = np.random.choice(np.arange(lod_ids_local.shape[0])[lod_occ_local],
                                                         size=n_points, replace=True)  # sample N local ids from LoD
                else:
                    return None
                lod_ids_local_rnd_filt = lod_ids_local[lod_ids_local_rnd]  # LoD local ids after occ filtering

                # Compute ids of the next level
                lod_next_ids = np.zeros((lod_ids_global.shape[0], 8)).astype(int)
                lod_next_ids[lod_ids_global_filt] = np.arange(lod_ids_global_filt.shape[0] * 8).reshape(-1, 8)
                lod_next_ids_rnd = lod_next_ids[lod_ids_local_rnd_filt].flatten()
                lod_ids_local_all.append(lod_next_ids_rnd)

                # Form output for the current level
                output[lod] = {}
                output[lod]['ids'] = lod_ids_local_rnd
                output[lod]['occ'] = lod_occ_local
                output[lod]['xyz'] = model[lod]['xyz'][lod_ids_local][lod_ids_local_rnd]
                output[lod]['sdf'] = np.expand_dims(model[lod]['sdf'][lod_ids_local][lod_ids_local_rnd], axis=-1)
                output[lod]['nrm'] = model[lod]['nrm'][lod_ids_local][lod_ids_local_rnd]

        return output
