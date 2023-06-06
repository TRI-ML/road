import os

import numpy as np
import open3d as o3d
import torch
from pytorch3d.loss import chamfer_distance
from torch.cuda.amp import autocast
from tqdm import tqdm

import data
from data.utils import to_cuda, collate_fn
from nets.orthanet import OrthaNet as ONet
from utils import io


def evaluate(cfg, onet=None, testset=None, feats=None, lod_current=None):
    """
    Evaluate trained network.

    Args:
        cfg: Configuration file
        onet: Pass network if already loaded
        testset: Pass existing dataset if already loaded
        feats: Pass features if already loaded
        lod_current: Octree level of detail used for evaluation

    Returns: Dictionary of metrics

    """
    # Prepare data
    if not testset:
        testset = data.get_dataset(cfg, 'testing')
    else:
        testset.test = True
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.cpu_threads,
                                             pin_memory=True, collate_fn=collate_fn)
    if lod_current:
        testset.lod_current = lod_current

    # Load model
    if onet is None:
        onet_dict = torch.load(os.path.join(cfg.path_output, 'onet.pt'))
        feats = onet_dict['feats']
        onet = ONet(cfg.latent_size, lods=cfg.lods, feat_combine=cfg.latent_combine, num_layers=cfg.num_layers,
                    hidden_dim=cfg.hidden_dim, decoder_layers=cfg.decoder_layers).to(cfg.device)
        onet.load_state_dict(onet_dict['model'], strict=False)
        print('Network restored!')
    onet.eval()

    # Metrics dict
    metrics = {
        'chamfer': [],
        'conf': []
    }

    lods = lod_current if lod_current else onet.lods

    # Training loop
    with autocast():
        with torch.no_grad():
            pbar = tqdm(enumerate(testloader), total=len(testloader))
            for i, gt in pbar:

                # Bring GT to device
                gt = to_cuda(gt, cfg.device)

                # Get output
                feat = feats['embedding'](gt['idx']).to(cfg.device).unsqueeze(0)
                pred_lod = onet.get_object(feat, lod_current=lods)

                # Project points onto the surface
                pred_pcd = pred_lod[-1]['xyz'] - (pred_lod[-1]['sdf'] * pred_lod[-1]['nrm'])
                pred_nrm = pred_lod[-1]['nrm']

                # Occupancy confidence of the last level
                occ_confidence = pred_lod[-1]['conf'].mean().item()
                metrics['conf'].append(occ_confidence)

                # Chamfer dist
                cd, _ = chamfer_distance(pred_pcd.float(), gt['pcd'].float())
                metrics['chamfer'].append(cd.item() * 1000)

                # Register losses
                log_str = 'Loss: '
                for text, val in metrics.items():
                    log_str += '{} - {:.6f}, '.format(text, val[-1])
                pbar.set_description(log_str)

                # Visualize
                if cfg.visualize:
                    # Assign new values to Open3D geometries
                    pcd_sdf_vis = o3d.geometry.PointCloud()
                    pcd_sdf_vis.points = o3d.utility.Vector3dVector(pred_pcd[0].detach().cpu())
                    pcd_sdf_vis.normals = o3d.utility.Vector3dVector(pred_nrm[0].detach().cpu())
                    pcd_sdf_vis.colors = o3d.utility.Vector3dVector(((pred_nrm[0] + 1) / 2).detach().cpu())

                    # Assign values to GT Open3D geometries
                    pcd_gt_vis = o3d.geometry.PointCloud()
                    pcd_gt_vis.points = o3d.utility.Vector3dVector(gt['pcd'][0].detach().cpu())
                    pcd_gt_vis.normals = o3d.utility.Vector3dVector(gt['nrm'][0].detach().cpu())
                    pcd_gt_vis.colors = o3d.utility.Vector3dVector(((gt['nrm'][0] + 1) / 2).detach().cpu())
                    pcd_gt_vis.translate([0, 2, 0])

                    # Update Open3D geometries
                    o3d.visualization.draw_geometries([pcd_sdf_vis, pcd_gt_vis])

            for k, v in metrics.items():
                metrics[k] = np.mean(v)

            log_str = 'AVG: '
            for text, val in metrics.items():
                log_str += '{} - {:.6f}, '.format(text, val)
            print(log_str)
            testset.test = False

            return metrics


def main():
    # Parse input
    args = io.parse_input()
    # Evaluate
    evaluate(args)


if __name__ == '__main__':
    main()
