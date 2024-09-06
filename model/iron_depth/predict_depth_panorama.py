import os
import numpy as np
from argparse import Namespace
import torch
from PIL import Image
from torchvision import transforms

import model.iron_depth.utils.utils as utils
from model.iron_depth.models_normal.NNET import NNET
from model.iron_depth.models.IronDepthPanorama import IronDepthPanorama

from equilib import equi2pers

from model.utils.utils import depth_to_distance_ratio


def load_iron_depth_model_panorama(type="scannet", iters=20, pano_iters=20, checkpoints_path="checkpoints", device="cuda:0"):
    args = Namespace(**{
        "train_iter": 3,
        "test_iter": iters,
        "test_pano_iter": pano_iters
    })

    if type == 'scannet':
        args.NNET_architecture = 'BN'
        args.NNET_ckpt = os.path.join(checkpoints_path, 'normal_scannet.pt')
        args.IronDepth_ckpt = os.path.join(checkpoints_path, 'irondepth_scannet.pt')
    elif type == 'nyuv2':
        args.NNET_architecture = 'GN'
        args.NNET_ckpt = os.path.join(checkpoints_path, 'normal_nyuv2.pt')
        args.IronDepth_ckpt = os.path.join(checkpoints_path, 'irondepth_nyuv2.pt')
    else:
        raise NotImplementedError("type=", type)

    device = torch.device(device)

    # define N_NET (surface normal estimation network)
    n_net = NNET(args).to(device)
    #print('loading N-Net weights from %s' % args.NNET_ckpt)
    n_net = utils.load_checkpoint(args.NNET_ckpt, n_net)
    n_net.eval()

    # define IronDepth
    model = IronDepthPanorama(args).to(device)
    #print('loading IronDepth weights from %s' % args.IronDepth_ckpt)
    model = utils.load_checkpoint(args.IronDepth_ckpt, model)
    model.eval()

    return n_net, model


def prepare_input(image: Image, K=None, device="cuda:0"):
    W, H = image.size
    device = torch.device(device)

    if K is None:
        K = np.eye(3).astype(np.float32)
        K[0, 0] = 500.0
        K[1, 1] = 500.0
        K[0, 2] = W // 2
        K[1, 2] = H // 2

    if isinstance(K, torch.Tensor):
        K = K.squeeze().detach().cpu().numpy()

    def get_pos(intrins, W, H):
        pos = np.ones((H, W, 2))
        x_range = np.concatenate([np.arange(W).reshape(1, W)] * H, axis=0)
        y_range = np.concatenate([np.arange(H).reshape(H, 1)] * W, axis=1)
        pos[:, :, 0] = x_range + 0.5
        pos[:, :, 1] = y_range + 0.5
        pos[:, :, 0] = np.arctan((pos[:, :, 0] - intrins[0, 2]) / intrins[0, 0])
        pos[:, :, 1] = np.arctan((pos[:, :, 1] - intrins[1, 2]) / intrins[1, 1])
        pos = torch.from_numpy(pos.astype(np.float32)).permute(2, 0, 1).to(device).unsqueeze(0)  # (1, 2, H, W)
        return pos

    # pos is an array of rays with unit depth
    pos = get_pos(K, W, H)

    img = np.array(image).astype(np.float32) / 255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img).permute(2, 0, 1)).to(device).unsqueeze(0)  # (1, 3, H, W)

    sample = {
        'img': img,
        'pos': pos
    }

    return sample


@torch.no_grad()
def predict_iron_depth(image, K=None, device="cuda:0", model=None, n_net=None, input_depth=None, input_mask=None, fix_input_depth=False):
    """
    Predict depth with IronDepth method.

    :param image: image from which to predict depth (PIL Image in range 0..255)
    :param K: intrinsics
    :param device: torch device
    :param model: iron_depth model
    :param n_net: iron_depth_n_net model
    :param input_depth: guidance depth that should be inpainted
    :param input_mask: where the guidance depth should be not trusted/inpainted. 1: these values should be inpainted. 0: these values should be trusted/fixed.
    :param fix_input_depth: if the guidance depth should remain unchanged or only initialized in the first iteration of iron_depth
    :return: predicted depth, predicted normal uncertainty
    """
    if model is None or n_net is None:
        model, n_net = load_iron_depth_model()
    
    data_dict = prepare_input(image, K, device)
    img = data_dict['img']
    pos = data_dict['pos']

    # surface normal prediction
    norm_out = n_net(img)
    pred_norm = norm_out[:, :3, :, :]
    pred_kappa = norm_out[:, 3:, :, :]
    alpha, alpha_deg = utils.kappa_to_alpha_torch(pred_kappa.permute(0, 2, 3, 1)[0, :, :, 0])

    input_dict = {
        'img': img,
        'pred_norm': pred_norm,
        'pred_kappa': pred_kappa,
        'pos': pos
    }

    # print(input_depth[~input_mask].max(), input_depth[~input_mask].min(), input_depth[~input_mask].mean())

    if input_depth is not None:
        input_dict['input_depth'] = input_depth.unsqueeze(0).unsqueeze(0)
        input_dict['input_mask'] = input_mask.unsqueeze(0).unsqueeze(0)
        input_dict['fix_input_depth'] = fix_input_depth

    # IronDepth forward pass
    pred_list = model.forward(input_dict, 'test')
    pred_depth = pred_list[-1].squeeze()

    return pred_depth, pred_norm.squeeze()


@torch.no_grad()
def predict_iron_depth_panorama(image, K=None, device="cuda:0", model=None, n_net=None, input_depth=None, input_mask=None, fix_input_depth=False, num_views=32):
    """
    Predict depth with IronDepth method.

    :param image: image from which to predict depth (PIL Image in range 0..255)
    :param K: intrinsics
    :param device: torch device
    :param model: iron_depth model
    :param n_net: iron_depth_n_net model
    :param input_depth: guidance depth that should be inpainted
    :param input_mask: where the guidance depth should be not trusted/inpainted. 1: these values should be inpainted. 0: these values should be trusted/fixed.
    :param fix_input_depth: if the guidance depth should remain unchanged or only initialized in the first iteration of iron_depth
    :return: predicted depth, predicted normal uncertainty
    """

    if model is None or n_net is None:
        model, n_net = load_iron_depth_model_panorama()

    input_dicts = []    
    views = [{'roll': 0, 'pitch': 0, 'yaw': -np.pi/num_views*2*i} for i in range(num_views)]
    to_pil = transforms.ToPILImage()
    dist_ratio = depth_to_distance_ratio(512, 512, K)
    pred_norms = [None for i in views]

    for i, view in enumerate(views):

        image_pers =to_pil(equi2pers(equi=image, rots=view, height=512, width=512, fov_x=90))
        inpaint_mask_pers = equi2pers(equi=input_mask.to(torch.float32).unsqueeze(0), rots=view, height=512, width=512, fov_x=90)
        inpaint_mask_pers[inpaint_mask_pers != 0] = 1
        inpaint_mask_pers = inpaint_mask_pers.to(torch.bool)
        
        depth_max = input_depth.max()
        depth_pers = equi2pers(equi=(input_depth/depth_max).unsqueeze(0), rots=view, height=512, width=512, fov_x=90)
        depth_pers = depth_pers * depth_max / dist_ratio

        data_dict = prepare_input(image_pers, K, device)
        img = data_dict['img']
        pos = data_dict['pos']
        

        # surface normal prediction
        norm_out = n_net(img)
        pred_norms[i] = norm_out[:, :3, :, :]
        pred_kappa = norm_out[:, 3:, :, :]
        alpha, alpha_deg = utils.kappa_to_alpha_torch(pred_kappa.permute(0, 2, 3, 1)[0, :, :, 0])

        input_dict = {
            'img': img,
            'pred_norm': pred_norms[i],
            'pred_kappa': pred_kappa,
            'pos': pos,
            'view': view,
            'K': K
        }


        if input_depth is not None:
            input_dict['input_depth'] = depth_pers.unsqueeze(0)
            input_dict['input_mask'] = inpaint_mask_pers.unsqueeze(0)
            input_dict['fix_input_depth'] = fix_input_depth

        input_dicts.append(input_dict)

    # IronDepth forward pass
    pred_list = model.forward_panorama(input_depth, input_mask, input_dicts, 'test')

    pred_depth = pred_list[-1].squeeze()

    data_dict = prepare_input(to_pil(image[..., 256:768, ]), K, device)
    pred_norm = torch.concat(pred_norms, dim=-1)


    return pred_depth, pred_norm.squeeze()
