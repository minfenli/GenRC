import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from model.iron_depth.models.submodules.DNET import DNET
from model.iron_depth.models.submodules.Dr_submodules import LSPN

import numpy as np
import cv2

from model.PerspectiveAndEquirectangular.lib import Equirec2Perspec as E2P
from model.PerspectiveAndEquirectangular.lib import multi_Perspec2Equirec as m_P2E

from model.depth_alignment import depth_alignment
from model.utils.utils import depth_to_distance_ratio
from tqdm import tqdm


# downsample the predicted norm, predicted kappa (surface normal confidence), and pos (ray with unit depth)
def downsample(input_dict, size):
    for k in ['pred_norm', 'pred_kappa', 'pos']:
        input_dict[k+'_down'] = F.interpolate(input_dict[k], size=size, mode='bilinear', align_corners=False)
        # make sure to normalize
        if k == 'pred_norm':
            norm = torch.sqrt(torch.sum(torch.square(input_dict[k+'_down']), dim=1, keepdim=True))
            norm[norm < 1e-10] = 1e-10
            input_dict[k+'_down'] = input_dict[k+'_down'] / norm
    
    if 'input_depth' in input_dict:
        input_dict['input_depth_down'] = F.interpolate(input_dict['input_depth'], size=size, mode='bilinear', align_corners=False)
    if 'input_mask' in input_dict:
        input_dict['input_mask_down'] = F.interpolate(input_dict['input_mask'].to(torch.float32), size=size, mode='bilinear', align_corners=False)
        input_dict['input_mask_down'][input_dict['input_mask_down'] != 0] = 1
        input_dict['input_mask_down'] = input_dict['input_mask_down'].to(torch.bool)
    if 'K' in input_dict:
        input_dict['K_down'] = input_dict['K'].clone()
        input_dict['K_down'][0, 0] = input_dict['K_down'][0, 0] / 8
        input_dict['K_down'][1, 1] = input_dict['K_down'][1, 1] / 8
        input_dict['K_down'][0, 2] = input_dict['K_down'][0, 2] / 8
        input_dict['K_down'][1, 2] = input_dict['K_down'][1, 2] / 8
    return input_dict


class IronDepthPanorama(nn.Module):
    def __init__(self, args):
        super(IronDepthPanorama, self).__init__()
        self.args = args
        self.downsample_ratio = 8

        # define D-Net
        self.output_dim = output_dim = 1
        self.feature_dim = feature_dim = 64
        self.hidden_dim = hidden_dim = 64

        #print('defining D-Net...')

        self.d_net = DNET(output_dims=[output_dim, feature_dim, hidden_dim])

        #print('defining Dr-Net...')
        self.dr_net = LSPN(args)

        self.ps = 5
        self.center_idx = (self.ps * self.ps - 1) // 2

        self.pad = (self.ps - 1) // 2
        self.irm_train_iter = args.train_iter             # 3
        self.irm_test_iter = args.test_iter               # 20
        self.irm_test_pano_iter = args.test_pano_iter               # 20

    def forward(self, input_dict, mode='train', ini_depth=None):
        pred_dmap, input_dict['feat'], h = self.d_net(input_dict['img'])

        down_size = [pred_dmap.size(2), pred_dmap.size(3)]
        input_dict = downsample(input_dict, down_size)

        if ini_depth is not None:
            pred_dmap = F.interpolate(ini_depth, size=down_size, mode='bilinear', align_corners=False)

        # depth_weights (B, ps*ps, H, W)
        input_dict['depth_candidate_weights'] = self.get_depth_candidate_weights(input_dict)

        # weights for upsampling
        input_dict['upsampling_weights'] = self.get_upsampling_weights(input_dict)

        # upsample first prediction
        up_pred_dmap = self.dr_net(h, pred_dmap, input_dict, upsample_only=True)

        up_pred_dmap = depth_alignment.scale_shift_linear(
                    rendered_depth=input_dict['input_depth'],
                    predicted_depth=up_pred_dmap,
                    mask=~input_dict['input_mask'],
                    fuse=False)

        if 'input_depth' in input_dict:
            # replace prediction from D-Net with other depth map, but use the same resolution as pred_dmap for consistency
            # only replace those pixels that are not masked out in input_mask
            mask = input_dict['input_mask'] > 0.5
            mask = mask + (input_dict['input_depth'].isinf()) + (input_dict['input_depth'] == 0)
            input_dict['input_mask'] = mask
            up_pred_dmap_merged = torch.where(mask, up_pred_dmap, input_dict['input_depth'])
            pred_dmap = F.interpolate(up_pred_dmap_merged, size=down_size, mode='bilinear', align_corners=False)

        # iterative refinement
        pred_list = [up_pred_dmap]
        N = self.irm_train_iter if mode == 'train' else self.irm_test_iter
        for i in range(N):
            h, pred_dmap, up_pred_dmap = self.dr_net(h, pred_dmap.detach(), input_dict)
            pred_list.append(up_pred_dmap)
            if 'input_depth' in input_dict and input_dict.get('fix_input_depth', False):
                # from section 5.2 in the paper: fix 'anchor points' from the input depth
                up_pred_dmap_merged = torch.where(input_dict['input_mask'], up_pred_dmap, input_dict['input_depth'])
                pred_dmap = F.interpolate(up_pred_dmap_merged, size=down_size, mode='bilinear', align_corners=False)

        return pred_list

    def forward_panorama(self, input_depth_pano, input_mask_pano, input_dicts, mode='train', ini_depth=None):
        pred_dmap = [None for i in input_dicts]
        h = [None for i in input_dicts]
        pred_dmaps = []
        up_pred_dmaps = []
        up_pred_dmaps_original = []
        views = []

        for i, input_dict in enumerate(input_dicts):
            views.append([90, -input_dict['view']['yaw']/np.pi*180, input_dict['view']['pitch']/np.pi*180])
            pred_dmap[i], input_dict['feat'], h[i] = self.d_net(input_dict['img'])

            down_size = [pred_dmap[i].size(2), pred_dmap[i].size(3)]
            input_dict = downsample(input_dict, down_size)

            if ini_depth is not None:
                pred_dmap[i] = F.interpolate(ini_depth, size=down_size, mode='bilinear', align_corners=False)

            # depth_weights (B, ps*ps, H, W)
            input_dict['depth_candidate_weights'] = self.get_depth_candidate_weights(input_dict)

            # weights for upsampling
            input_dict['upsampling_weights'] = self.get_upsampling_weights(input_dict)

            # upsample first prediction
            up_pred_dmap = self.dr_net(h[i], pred_dmap[i], input_dict, upsample_only=True)
            #up_pred_dmap[up_pred_dmap <= 0] = 0.1
            #'''
            up_pred_dmaps_original.append(up_pred_dmap)
            up_pred_dmaps.append((up_pred_dmap.squeeze(1) * depth_to_distance_ratio(512, 512, input_dict['K'])).to(torch.float32).cpu().numpy())
            
        per = m_P2E.Perspective(up_pred_dmaps, views)
        up_depth_pano = torch.tensor(per.GetEquirec(512*2, 512*4, 1), device=up_pred_dmap.device, dtype = up_pred_dmap.dtype).squeeze(0)
        #print(up_depth_pano.mean())
        input_mask_pano[up_depth_pano == 0] = True
        
        up_depth_pano = depth_alignment.scale_shift_linear(
            rendered_depth=input_depth_pano,
            predicted_depth=up_depth_pano,
            mask=~input_mask_pano,
            fuse=False)
        
        equ = E2P.Equirectangular(up_depth_pano.unsqueeze(0).to(torch.float32).cpu().numpy())


        for i, input_dict in enumerate(input_dicts):
            if (~input_dict['input_mask']).sum() == 0:
                up_pred_dmap = torch.tensor(equ.GetPerspective(90, -input_dict['view']['yaw']/np.pi*180, input_dict['view']['pitch']/np.pi*180, 512, 512, 1), device=up_pred_dmap.device, dtype = up_pred_dmap.dtype)   #(FOV, theta, phi, height, width)
                up_pred_dmap =  up_pred_dmap / depth_to_distance_ratio(512, 512, input_dict['K'])
            else:
                up_pred_dmap = up_pred_dmaps_original[i]
                up_pred_dmap = depth_alignment.scale_shift_linear(
                    rendered_depth=input_dict['input_depth'],
                    predicted_depth=up_pred_dmap,
                    mask=~input_dict['input_mask'],
                    fuse=False)

            if 'input_depth' in input_dict:
                # replace prediction from D-Net with other depth map, but use the same resolution as pred_dmap for consistency
                # only replace those pixels that are not masked out in input_mask
                mask = input_dict['input_mask'] > 0.5
                mask = mask + (input_dict['input_depth'].isinf()) + (input_dict['input_depth'] == 0)
                input_dict['input_mask'] = mask
                up_pred_dmap_merged = torch.where(mask, up_pred_dmap, input_dict['input_depth'])
                pred_dmap[i] = F.interpolate(up_pred_dmap_merged, size=down_size, mode='bilinear', align_corners=False)

            
        
        # iterative refinement
        
        pred_list = []
        N = self.irm_train_iter if mode == 'train' else self.irm_test_pano_iter
        for iron_iter in tqdm(range(N)):
            up_pred_dmaps = []

            for i, input_dict in enumerate(input_dicts):
                h[i], pred_dmap[i], up_pred_dmap = self.dr_net(h[i], pred_dmap[i].detach(), input_dict)
                up_pred_dmaps.append((up_pred_dmap.squeeze(1) * depth_to_distance_ratio(512, 512, input_dict['K'])).to(torch.float32).cpu().numpy())
            
            #global averaging stuff
            for i, input_dict in enumerate(input_dicts):
                if 'input_depth' in input_dict and input_dict.get('fix_input_depth', False):
                    #warp every other views to current view
                    ones = np.ones_like(up_pred_dmaps[i][0, ...])
                    depth_pers_value = np.zeros_like(up_pred_dmaps[i][0, ...])
                    depth_pers_count = np.zeros_like(up_pred_dmaps[i][0, ...])
                    
                    for j, input_dict_other in enumerate(input_dicts):
                        if self.radians_difference_less_than_90_deg(input_dict['view']['yaw'], input_dict_other['view']['yaw']) and self.radians_difference_less_than_90_deg(input_dict['view']['pitch'], input_dict_other['view']['pitch']):
                            K,      R      = self.get_K_R(90, input_dict_other['view']['yaw']/np.pi*180, -input_dict_other['view']['pitch']/np.pi*180, 512, 512)
                            self_K, self_R = self.get_K_R(90, input_dict['view']['yaw']/np.pi*180, -input_dict['view']['pitch']/np.pi*180, 512, 512)
                            H = self_K @ self_R @ R.T @ np.linalg.inv(K)
                            weight_factor = 1
                            if (~input_dict_other['input_mask']).sum() == 0:
                                weight_factor = 0.25
                            depth_pers_value += cv2.warpPerspective(np.transpose(up_pred_dmaps[j], (1, 2, 0)), H, (512, 512), flags=cv2.INTER_LINEAR) * weight_factor
                            depth_pers_count += cv2.warpPerspective(ones, H, (512, 512), flags=cv2.INTER_LINEAR) * weight_factor
                    
                    up_pred_dmap = torch.tensor(np.where(depth_pers_count, depth_pers_value / depth_pers_count, 0), device=pred_dmap[0].device).unsqueeze(0)
                    up_pred_dmap = up_pred_dmap / depth_to_distance_ratio(512, 512, input_dict['K'])

                    # from section 5.2 in the paper: fix 'anchor points' from the input depth
                    up_pred_dmap = torch.where(input_dict['input_mask'], up_pred_dmap, input_dict['input_depth'])
                    pred_dmap[i] = F.interpolate(up_pred_dmap, size=down_size, mode='bilinear', align_corners=False)

        per = m_P2E.Perspective(up_pred_dmaps, views)
        final_depth_pano = torch.tensor(per.GetEquirec(512*2, 512*4, 1), device=up_pred_dmap.device, dtype = up_pred_dmap.dtype)    
        return [final_depth_pano.unsqueeze(1)]

    def get_K_R(self, FOV, THETA, PHI, height, width):
        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0,  1],
        ], np.float32)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        return K, R

    def radians_difference_less_than_90_deg(self, angle_1, angle_2):
        '''
        angle_1 and angle_2 are in np type and are radians
        '''
        difference = np.abs((angle_1 / np.pi * 180) % 360 - (angle_2 / np.pi * 180) % 360)
        return difference < 90 or difference > 270

    def get_depth_candidate_weights(self, input_dict):
        with torch.no_grad():
            B, _, H, W = input_dict['pred_norm_down'].shape

            # pred norm down - nghbr
            pred_norm_down = F.pad(input_dict['pred_norm_down'], pad=(self.pad, self.pad, self.pad, self.pad), mode='replicate')
            pred_norm_down_unfold = F.unfold(pred_norm_down, [self.ps, self.ps], padding=0)                                     # (B, 3*ps*ps, H*W)
            pred_norm_down_unfold = pred_norm_down_unfold.view(B, 3, self.ps*self.ps, H, W)                                     # (B, 3, ps*ps, H, W)

            # pos down - nghbr
            pos_down_nghbr = F.pad(input_dict['pos_down'], pad=(self.pad, self.pad, self.pad, self.pad), mode='replicate')
            pos_down_nghbr_unfold = F.unfold(pos_down_nghbr, [self.ps, self.ps], padding=0)                                         # (B, 2*ps*ps, H*W)
            pos_down_nghbr_unfold = pos_down_nghbr_unfold.view(B, 2, self.ps*self.ps, H, W)                                         # (B, 2, ps*ps, H, W)

            # norm and pos - nghbr
            nx, ny, nz = pred_norm_down_unfold[:, 0, ...], pred_norm_down_unfold[:, 1, ...], pred_norm_down_unfold[:, 2, ...]       # (B, ps*ps, H, W) or (B, 1, H, W)
            pos_u, pos_v = pos_down_nghbr_unfold[:, 0, ...], pos_down_nghbr_unfold[:, 1, ...]                                       # (B, ps*ps, H, W)

            # pos - center
            pos_u_center = pos_u[:, self.center_idx, :, :].unsqueeze(1)                                                             # (B, 1, H, W)
            pos_v_center = pos_v[:, self.center_idx, :, :].unsqueeze(1)                                                             # (B, 1, H, W)

            ddw_num = nx * pos_u + ny * pos_v + nz
            ddw_denom = nx * pos_u_center + ny * pos_v_center + nz
            ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8

            ddw_weights = ddw_num / ddw_denom                                                                                       # (B, ps*ps, H, W)
            ddw_weights[ddw_weights != ddw_weights] = 1.0               # nan
            ddw_weights[torch.abs(ddw_weights) == float("Inf")] = 1.0   # inf
        return ddw_weights

    def get_upsampling_weights(self, input_dict):
        with torch.no_grad():
            B, _, H, W = input_dict['pos_down'].shape
            k = self.downsample_ratio

            # norm nghbr
            pred_norm_down = F.pad(input_dict['pred_norm_down'], pad=(1,1,1,1), mode='replicate')
            up_norm = F.unfold(pred_norm_down, [3, 3], padding=0)   # (B, 3, H, W) -> (B, 3 X 3*3, H*W)
            up_norm = up_norm.view(B, 3, 9, 1, 1, H, W)             # (B, 3, 3*3, 1, 1, H, W)

            # pos nghbr
            pos_down = F.pad(input_dict['pos_down'], pad=(1,1,1,1), mode='replicate')
            up_pos_nghbr = F.unfold(pos_down, [3, 3], padding=0)        # (B, 2, H, W) -> (B, 2 X 3*3, H*W)
            up_pos_nghbr = up_pos_nghbr.view(B, 2, 9, 1, 1, H, W)       # (B, 2, 3*3, 1, 1, H, W)

            # pos ref
            pos = input_dict['pos']
            up_pos_ref = pos.reshape(B, 2, H, k, W, k)                  # (B, 2, H, k, W, k)
            up_pos_ref = up_pos_ref.permute(0, 1, 3, 5, 2, 4)           # (B, 2, k, k, H, W)
            up_pos_ref = up_pos_ref.unsqueeze(2)                        # (B, 2, 1, k, k, H, W)

            # compute new depth
            new_depth_num = (up_norm[:, 0:1, ...] * up_pos_nghbr[:, 0:1, ...]) + \
                            (up_norm[:, 1:2, ...] * up_pos_nghbr[:, 1:2, ...]) + \
                            (up_norm[:, 2:3, ...])                      # (B, 1, 3*3, 1, 1, H, W)

            new_depth_denom = (up_norm[:, 0:1, ...] * up_pos_ref[:, 0:1, ...]) + \
                              (up_norm[:, 1:2, ...] * up_pos_ref[:, 1:2, ...]) + \
                              (up_norm[:, 2:3, ...])                    # (B, 1, 3*3, k, k, H, W)

            new_depth_denom[torch.abs(new_depth_denom) < 1e-8] = torch.sign(new_depth_denom[torch.abs(new_depth_denom) < 1e-8]) * 1e-8
            new_depth = new_depth_num / new_depth_denom                 # (B, 1, 3*3, k, k, H, W)

            # check for nan, inf
            new_depth[new_depth != new_depth] = 1.0  # nan
            new_depth[torch.abs(new_depth) == float("Inf")] = 1.0  # inf        
        return new_depth
