import torch
from torch import nn

from nets.siren import SirenONet
from data.utils import get_cell_size, subdivide


class OrthaNet(nn.Module):
    """ Main network class """

    def __init__(self, latent_size=64, lods=7, hidden_dim=1024, num_layers=4, feat_combine='reduce', decoder_layers=0):
        super(OrthaNet, self).__init__()
        self.lat_size = latent_size
        self.feat_combine = feat_combine

        self.lods = lods
        self.lodnet = nn.ModuleList()

        # Create a base network for each LoD
        # If concatenation is used, we need to create a network for each LoD
        if feat_combine == 'concatenate':
            for l in range(lods + 1):
                self.lodnet.append(SirenONet(
                    dim_in=self.lat_size * (l + 1),  # input dimension, ex. 2d coor
                    dim_hidden=hidden_dim,  # hidden dimension
                    dim_out=self.lat_size,  # output dimension, ex. rgb value
                    num_layers=num_layers,  # number of layers
                    final_activation=nn.Identity(),  # activation of final layer (nn.Identity() for direct output)
                    w0_initial=30.
                    # different signals may require different omega_0 in the first layer - this is a hyperparameter
                ))
        else:
            # If none or reduction is used, we only need one network
            self.lodnet_base = SirenONet(
                dim_in=self.lat_size,  # input dimension, ex. 2d coor
                dim_hidden=hidden_dim,
                dim_out=self.lat_size,  # output dimension, ex. rgb value
                num_layers=num_layers,  # number of layers
                final_activation=nn.Identity(),  # activation of final layer (nn.Identity() for direct output)
                decoder_layers=decoder_layers,
                w0_initial=30.
                # different signals may require different omega_0 in the first layer - this is a hyperparameter
            )
            for _ in range(lods + 1):
                self.lodnet.append(self.lodnet_base)

    def combine(self, feat1, feat2):
        """ Combine two features based on the defined feature fusion method """
        if self.feat_combine == 'concatenate':
            return torch.cat([feat1, feat2], dim=-1)
        elif self.feat_combine == 'reduce':
            return feat1 + feat2
        elif self.feat_combine == 'none':
            return feat2

    def forward(self, feats, gt, lod_current=None):
        """
        Extract objects from a batch of features

        Args:
            feats: Feature embedding
            gt: GT data dict
            lod_current: Octree level of detail

        Returns: Dict of extracted features (lat) and regressed properties (sdf, nrm, occ, xyz) at each LoD

        """
        # First LoD
        pred_lod = []
        pred = {}
        pred['lat'] = feats[:, None]
        pred['xyz'] = torch.zeros(pred['lat'].shape[0], 1, 3).to(gt[0]['xyz'].device)
        pred['occ'] = torch.tensor([[[0., 1.]]]).repeat(pred['lat'].shape[0], 1, 1).to(gt[0]['xyz'].device)
        pred_lod.append(pred)

        # Traverse through further LoDs to extract object
        lods = lod_current if lod_current else self.lods
        for lod in range(lods):
            pred = {}

            lat, pred['occ'], sdf, nrm = self.lodnet[lod](pred_lod[lod]['lat'])
            pred['lat'] = torch.stack([(self.combine(pred_lod[lod]['lat'].repeat_interleave(8, dim=-2), lat))[
                                           k].index_select(0, gt[lod + 1]['ids'][k])
                                       for k in torch.arange(gt[0]['xyz'].shape[0])])
            pred['xyz'] = torch.stack(
                [subdivide(pred_lod[lod]['xyz'], level=lod)[k].index_select(0, gt[lod + 1]['ids'][k]) for k in
                 torch.arange(gt[0]['xyz'].shape[0])])
            pred['sdf'] = torch.stack([sdf[k].index_select(0, gt[lod + 1]['ids'][k]) for k in
                                       torch.arange(gt[0]['xyz'].shape[0])]) * get_cell_size(lod + 1)
            pred['nrm'] = torch.stack(
                [nrm[k].index_select(0, gt[lod + 1]['ids'][k]) for k in torch.arange(gt[0]['xyz'].shape[0])])

            pred_lod.append(pred)

        return pred_lod

    def get_object(self, feat, lod_current=None):
        """
        Extract an object from a single object feature

        Args:
            feat: Single feature
            lod_current: Octree level of detail

        Returns: Dict of extracted features (lat) and regressed properties (sdf, nrm, occ, conf, xyz) at each LoD

        """
        # First LoD
        pred_lod = []
        pred = {}
        pred['lat'] = feat.unsqueeze(0)
        pred['xyz'] = torch.zeros(pred['lat'].shape[0], 1, 3).to(feat.device)
        pred['occ'] = torch.tensor([[[0., 1.]]]).repeat(pred['lat'].shape[0], 1, 1).to(feat.device)
        pred_lod.append(pred)

        # Traverse through further LoDs to extract object
        lods = lod_current if lod_current else self.lods
        for lod in range(lods):
            pred = {}
            lat, occ, sdf, nrm = self.lodnet[lod](pred_lod[lod]['lat'])
            pred['occ'] = occ.softmax(-1).max(-1, keepdims=False).indices.bool()
            pred['conf'] = occ.softmax(-1).max(-1, keepdims=False).values
            pred['lat'] = (self.combine(pred_lod[lod]['lat'].repeat_interleave(8, dim=-2),
                                        lat))[pred['occ']].unsqueeze(0)
            pred['xyz'] = subdivide(pred_lod[lod]['xyz'], level=lod)[pred['occ']].view(pred['occ'].shape[0], -1, 3)
            pred['sdf'] = sdf[pred['occ']].view(pred['occ'].shape[0], -1, 1) * get_cell_size(lod + 1)
            pred['nrm'] = nrm[pred['occ']].view(pred['occ'].shape[0], -1, 3)
            pred_lod.append(pred)

        return pred_lod
