import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import data
import wandb
from data.utils import to_cuda, get_cell_size, collate_fn
from evaluate import evaluate
from nets.orthanet import OrthaNet as ONet
from utils import io


def train(cfg):
    """ Main training function """
    # Wandb Logger
    wandb.init(project=cfg.wandb, entity='tri', mode=os.getenv('WANDB_MODE', 'run'),
               config=cfg)
    cfg = wandb.config

    # Prepare data
    trainset = data.get_dataset(cfg, 'training')
    if cfg.curriculum:
        trainset.lod_current = cfg.curriculum
    else:
        trainset.lod_current = cfg.lods
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True,
                                              num_workers=cfg.cpu_threads, pin_memory=True, collate_fn=collate_fn)

    # Load model
    onet = ONet(cfg.latent_size, lods=cfg.lods, feat_combine=cfg.latent_combine, num_layers=cfg.num_layers,
                hidden_dim=cfg.hidden_dim, decoder_layers=cfg.decoder_layers).to(
        cfg.device)

    # Create a feature per object
    feats = {}
    num_models = len(trainset.models)
    feats['idx'] = [m['idx'] for m in trainset.models]
    feats['embedding'] = nn.Embedding(num_models, cfg.latent_size).to(cfg.device)
    torch.nn.init.normal_(feats['embedding'].weight.data, 0, 1 / math.sqrt(cfg.latent_size))

    # Optimizer
    optimizer = optim.Adam(
        [
            {'params': onet.parameters(), 'lr': cfg.learning_rate},
            {'params': feats['embedding'].parameters(), 'lr': cfg.learning_rate_latent}
        ],
        lr=cfg.learning_rate)

    # Recover model and features
    if cfg.path_net:
        onet_dict = torch.load(os.path.join(cfg.path_net, 'onet.pt'))
        onet.load_state_dict(onet_dict['model'], strict=False)
        optimizer.load_state_dict(onet_dict['optimizer'])
        feats['embedding'] = nn.Embedding.from_pretrained(onet_dict['feats']['embedding'].weight)
        feats['idx'] = onet_dict['feats']['idx']
        print('Network restored!')

    # Losses
    loss_dict = {}
    loss_ce = torch.nn.CrossEntropyLoss()
    score_best = float('inf')

    # Scaler and scheduler
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.scheduler_step, gamma=cfg.scheduler_decay
    )

    # Training
    for epoch in range(cfg.epochs_max):
        onet.train()

        # Training loop
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for i, gt in pbar:

            with autocast():

                # Bring GT to device
                gt = to_cuda(gt, cfg.device)

                # Zero gradients
                optimizer.zero_grad()

                # Get output
                feats_batch = feats['embedding'](gt['idx'])
                pred_lod = onet(feats_batch, gt, lod_current=trainset.lod_current)

                # Compute losses
                losses_occ, losses_sdf, losses_nrm, losses_rgb = [], [], [], []
                for lod in range(1, trainset.lod_current + 1):
                    losses_occ.append(
                        loss_ce(pred_lod[lod]['occ'].permute(0, 2, 1), gt[lod]['occ'].long()))  # Occupancy loss
                    losses_sdf.append(
                        (pred_lod[lod]['sdf'] - gt[lod]['sdf']).
                        norm(p=2, dim=-1).mean() / get_cell_size(lod))  # SDF loss
                    losses_nrm.append(
                        (pred_lod[lod]['nrm'] - gt[lod]['nrm']).norm(p=2, dim=-1).mean())  # Surface normals loss

                loss_dict['OCC'] = torch.stack(losses_occ).mean() * cfg.w_occ
                loss_dict['SDF'] = torch.stack(losses_sdf).mean() * cfg.w_sdf
                loss_dict['NRM'] = torch.stack(losses_nrm).mean() * cfg.w_nrm
                # Combine losses
                loss = sum(loss_dict.values())

            # Backward computation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Register losses (W&B)
            log_str = 'Epoch {}, Loss: '.format(epoch)
            for text, val in loss_dict.items():
                log_str += '{} - {:.6f}, '.format(text, val)
                # W&B logger
                if i % cfg.iter_log == 0:
                    wandb.log({text: val})
            wandb.log({'LR': scheduler.get_last_lr()[0]})
            pbar.set_description(log_str)

        # Store model
        if epoch > 0 and epoch % cfg.epoch_analyze == 0:

            # Validation
            metrics = evaluate(cfg, onet=onet, testset=trainset, feats=feats, lod_current=trainset.lod_current)
            wandb.log({'Chamfer': metrics['chamfer']})
            wandb.log({'Confidence': metrics['conf']})
            wandb.log({'LoD': trainset.lod_current})

            if metrics['chamfer'] < score_best:
                sv_file = {
                    'model': onet.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'feats': feats,
                }
                if cfg.path_output:
                    os.makedirs(cfg.path_output, exist_ok=True)
                    torch.save(sv_file, os.path.join(cfg.path_output, 'onet.pt'))
                else:
                    torch.save(sv_file, os.path.join(wandb.run.dir, 'onet.pt'))
                score_best = metrics['chamfer']

            # Level switcher
            if metrics['conf'] > cfg.conf_thres and trainset.lod_current < cfg.lods:
                trainset.lod_current += 1
                print('Welcome to level {}!'.format(trainset.lod_current))


def main():
    # Parse input
    args = io.parse_input()

    # Save config
    os.makedirs(args.path_output, exist_ok=True)
    with open(os.path.join(args.path_output, 'cfg.yaml'), 'w') as yamlfile:
        yaml.dump(vars(args), yamlfile)
        yamlfile.close()
        print("Config saved")

    # Start training
    train(args)


if __name__ == '__main__':
    main()
