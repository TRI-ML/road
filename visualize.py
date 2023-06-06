import os

import open3d as o3d
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from nets.orthanet import OrthaNet as ONet
from utils import io


def visualize(cfg):
    """ Visualize trained network """
    # Load model
    onet_dict = torch.load(os.path.join(cfg.path_output, 'onet.pt'))
    onet = ONet(cfg.latent_size, lods=cfg.lods, feat_combine=cfg.latent_combine,
                num_layers=cfg.num_layers, hidden_dim=cfg.hidden_dim, decoder_layers=cfg.decoder_layers).to(cfg.device)
    onet.load_state_dict(onet_dict['model'], strict=False)
    onet.eval()

    # Latent vectors
    feats = onet_dict['feats']
    obj_ids = feats['idx']
    obj_feats = feats['embedding'].to(cfg.device)

    # Loop over latent vectors
    with torch.no_grad():
        with autocast():
            pbar = tqdm(enumerate(obj_ids))
            for i, idx in pbar:
                # Get output
                pred_lod = onet.get_object(obj_feats(torch.tensor(i).to(cfg.device)))

                # Project points onto the surface
                pred_pcd = pred_lod[-1]['xyz'] - (pred_lod[-1]['sdf'] * pred_lod[-1]['nrm'])
                pred_nrm = pred_lod[-1]['nrm']

                # Visualize using Open3D
                pcd_sdf_vis = o3d.geometry.PointCloud()
                pcd_sdf_vis.points = o3d.utility.Vector3dVector(pred_pcd[0].detach().cpu())
                pcd_sdf_vis.normals = o3d.utility.Vector3dVector(pred_nrm[0].detach().cpu())
                pcd_sdf_vis.colors = o3d.utility.Vector3dVector(((pred_nrm[0] + 1) / 2).detach().cpu())
                o3d.visualization.draw_geometries([pcd_sdf_vis], width=600, height=600,
                                                  window_name='Object: {}'.format(idx))


def main():
    # Parse input
    args = io.parse_input()
    # Visualize
    visualize(args)


if __name__ == '__main__':
    main()
