import cv2
import numpy as np
import torch
from PIL import Image
import json
import os
from model.utils.utils import save_image, save_image_pair, visualize_depth_numpy
from model.mesh_fusion.util import get_pinhole_intrinsics_from_fov
import math
from model.genrc_pipeline import GenRCPipeline

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
import clip
from tqdm import tqdm
from utils.matrix_interp import matrix_interpolate

from pytorch3d.ops import sample_points_from_meshes, knn_points
#from pytorch3d.loss import chamfer_distance
from evaluate.chamfer import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

import copy

def calculate_masked_mse(gt, predict, mask):
    # calculate the difference between gt_depth and predicted_depth excluding pixels where mask==1
    if (~mask).sum()==0:
        return 0.
    
    return ((gt - predict)[~mask]**2).mean()

# def calc_mse_psnr(gt, predict, mask, skip=2):
#     """
#     Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
#     """
#     mse = calc_mse(gt, predict, mask)
#     psnr = -10.0 * np.log10(mse)

#     return mse, psnr

def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# --------------------------------------------
# LPIPS
# --------------------------------------------
def calculate_lpips(img1, img2, device="cuda"):
    '''calculate LPIPS
        img1, img2: [0, 255]
    '''

    img1 = torch.FloatTensor(np.array(img1)).to(device) / 255.0 * 2 - 1
    img2 = torch.FloatTensor(np.array(img2)).to(device)  / 255.0 * 2 - 1

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device) # input shape: (N, 3, H, W)

    if img1.ndim == 3:
        if img1.shape[2] == 3:
            # (H, W, 3) -> (3, H, W)
            img1 = torch.permute(img1, (2, 0, 1))[None, ...]
            img2 = torch.permute(img2, (2, 0, 1))[None, ...]
    elif img1.ndim == 4:
        if img1.shape[3] == 3:
            # (N, H, W, 3) -> (N, 3, H, W)
            img1 = torch.permute(img1, (0, 3, 1, 2))
            img2 = torch.permute(img2, (0, 3, 1, 2))
    else:
        raise ValueError('Wrong input image dimensions.')
    
    return lpips(img1, img2)
    

# --------------------------------------------
# IS (Inception Score)
# --------------------------------------------
def calculate_is(img, device="cuda"):
    '''calculate IS
        img: [0, 255]
    '''

    img = torch.tensor(np.array(img), dtype=torch.uint8).to(device) 

    inc_score = InceptionScore().to(device)  # input shape: (N, 3, H, W)

    # print(img, img.shape)

    if img.ndim == 3:
        if img.shape[2] == 3:
            # (H, W, 3) -> (3, H, W)
            img = torch.permute(img, (2, 0, 1))[None, ...]
    elif img.ndim == 4:
        if img.shape[3] == 3:
            # (N, H, W, 3) -> (N, 3, H, W)
            img = torch.permute(img, (0, 3, 1, 2))
    else:
        raise ValueError('Wrong input image dimensions.')
    
    # print(img, img.shape)

    inc_score.update(img)
    mean, std = inc_score.compute()

    return mean, std


# --------------------------------------------
# FID (Frechet Inception Distance)
# --------------------------------------------
def calculate_fid(img1, img2, device="cuda"):
    '''calculate LPIPS
        img1(initial or ground-truth images), img2(generated): [0, 255]
    '''
    
    img1 = torch.tensor(np.array(img1), dtype=torch.uint8).to(device) 
    img2 = torch.tensor(np.array(img2), dtype=torch.uint8).to(device) 

    # initial images may be fewer than generated images
    if not img1.shape[1:] == img2.shape[1:]:
        raise ValueError('Input images must have the same dimensions.')
    
    fid = FrechetInceptionDistance(feature=64).to(device) 

    if img1.ndim == 3:
        if img1.shape[2] == 3:
            # (H, W, 3) -> (3, H, W)
            img1 = torch.permute(img1, (2, 0, 1))[None, ...]
            img2 = torch.permute(img2, (2, 0, 1))[None, ...]
    elif img1.ndim == 4:
        if img1.shape[3] == 3:
            # (N, H, W, 3) -> (N, 3, H, W)
            img1 = torch.permute(img1, (0, 3, 1, 2))
            img2 = torch.permute(img2, (0, 3, 1, 2))
    else:
        raise ValueError('Wrong input image dimensions.')
    
    fid.update(img1, real=True)
    fid.update(img2, real=False)
    
    return fid.compute()


# --------------------------------------------
# CLIP Score
# --------------------------------------------
def calculate_cs(img1, img2, device="cuda"):
    '''calculate CLIP Score
        img1(initial or ground-truth images), img2(generated): [0, 255]
    '''

    model, preprocess = clip.load("ViT-B/32", device=device)

    with torch.no_grad():
        img1_clip_features = []
        for img in img1:
            image = preprocess(Image.fromarray(img.astype(np.uint8))).unsqueeze(0).to(device)
            features = model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
            img1_clip_features.append(features)
        img1_clip_feature_mean = torch.stack(img1_clip_features).mean(axis=0)
        img1_clip_feature_mean /= img1_clip_feature_mean.norm(dim=-1, keepdim=True)

        img2_clip_features = []
        for img in img2:
            image = preprocess(Image.fromarray(img.astype(np.uint8))).unsqueeze(0).to(device)
            features = model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
            img2_clip_features.append(features)
        img2_clip_feature_mean = torch.stack(img2_clip_features).mean(axis=0)
        img2_clip_feature_mean /= img2_clip_feature_mean.norm(dim=-1, keepdim=True)
        cs = img1_clip_feature_mean @ img2_clip_feature_mean.T
    
    return cs.item()


def resize(image, H=128, W=128):
    assert len(image.shape) == 2 or image.shape[-1] == 3
    if len(image.shape) == 2 and image.dtype == 'bool':
        image = cv2.resize(image.astype('uint8'), (W, H), interpolation=cv2.INTER_AREA)
        image = image!=0
    else: 
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
    return image

def init_scannet(pipeline: GenRCPipeline, init_dataset, args, offset=0):
    first_image_pils = []
    poses = []
    intrinsics = []
    gt_depths = []
    inpaint_masks = []
    w2cs = []
    
    for data in init_dataset:
        image_pil = Image.fromarray(data['image'])
        w2c= torch.tensor(data['w2c'], device=args.device)
        gt_depth = torch.tensor(data['depth'], device=args.device)
        intrinsic = torch.tensor(data['K'], device=args.device)
        inpaint_mask = torch.tensor(~data['instance_mask'], device=args.device)
        w2cs.append(data['w2c'])

        first_image_pils.append(image_pil)
        poses.append(w2c)
        intrinsics.append(intrinsic)
        gt_depths.append(gt_depth)
        inpaint_masks.append(inpaint_mask)

    offset = pipeline.init_with_posed_images(first_image_pils, poses=poses, intrinsics=intrinsics, gt_depths=gt_depths, inpaint_masks=inpaint_masks, offset=0)

    # use a bigger or smaller fov if given fov
    if args.inpaint_initialized_images_with_fov is not None:
        offset = inpaint_with_poses(pipeline, w2cs, offset, args.inpaint_initialized_images_with_fov)

    return offset

def inpaint_with_poses(pipeline: GenRCPipeline, w2cs, offset=0, fov=None):
    
    # use a bigger or smaller fov if given fov
    if fov is not None:
        old_intrinsic = pipeline.K
        pipeline.K = get_pinhole_intrinsics_from_fov(H=pipeline.H, W=pipeline.W, fov_in_degrees=fov).to(pipeline.world_to_cam)


    # generate poses using unsampled poses in the scannet scene
    pipeline.set_simple_trajectory_poses(w2cs)
    offset = pipeline.generate_images_with_poses(offset=offset)

    if fov is not None:
        pipeline.K = old_intrinsic

    return offset 

def render_with_poses(pipeline: GenRCPipeline, w2cs, fov=None, n_iterp=5, rotate=False, depth_minmax=None):
    
    # use a bigger or smaller fov if given fov
    if fov is not None:
        old_intrinsic = pipeline.K
        pipeline.K = get_pinhole_intrinsics_from_fov(H=pipeline.H, W=pipeline.W, fov_in_degrees=fov).to(pipeline.world_to_cam)

    rendered_image_pils = []
    rendered_depth_pils = []
    inpaint_mask_pils = []
    rendered_rgbd_pils =[]

    pbar = tqdm(range(len(w2cs)))
    for pos in pbar:
        pbar.set_description(f"Image [{pos}/{len(w2cs)-1}]")
        rendered_image_pil, \
        rendered_depth_pil, \
        inpaint_mask_pil, \
        rendered_rgbd_pil = pipeline.render_with_pose_pil(pipeline.K, w2cs[pos], rotate, depth_minmax)

        rendered_image_pils.append(rendered_image_pil)
        rendered_depth_pils.append(rendered_depth_pil)
        inpaint_mask_pils.append(inpaint_mask_pil)
        rendered_rgbd_pils.append(rendered_rgbd_pil)

        if n_iterp > 0 and pos < len(w2cs) -1 :
            fracs = torch.linspace(0, 1, steps=n_iterp+2)[1:-1]
            for frac in fracs:
                pos_interp = matrix_interpolate(w2cs[pos], w2cs[pos+1], frac)

                rendered_image_pil, \
                rendered_depth_pil, \
                inpaint_mask_pil, \
                rendered_rgbd_pil = pipeline.render_with_pose_pil(pipeline.K, pos_interp, rotate, depth_minmax)

                rendered_image_pils.append(rendered_image_pil)
                rendered_depth_pils.append(rendered_depth_pil)
                inpaint_mask_pils.append(inpaint_mask_pil)
                rendered_rgbd_pils.append(rendered_rgbd_pil)

    if fov is not None:
        pipeline.K = old_intrinsic
    
    return rendered_image_pils, rendered_depth_pils, inpaint_mask_pils, rendered_rgbd_pils

def inpaint_panorama(pipeline: GenRCPipeline, init_dataset, args, offset=0):
    depth_mse = compute_depth_mse(pipeline, init_dataset, args, H=480, W=640)
    f_decision = open(os.path.join(args.out_path, "rgb", "decision.txt"), 'w')
    f_decision.write(f"Input Depth MSE before panorama: {depth_mse}\n\n")

    pano_pipeline = None
    min_depth_mse = -1
    decision = 0
    
    center_pos_list = get_center_pos_list(args, init_dataset)

    init_vertices, init_colors, init_faces = pipeline.vertices, pipeline.colors, pipeline.faces

    for i, center_pos in enumerate(center_pos_list):
        pipeline.vertices, pipeline.colors, pipeline.faces = init_vertices, init_colors, init_faces
        offset = pipeline.project_and_inpaint_panorama(center_pos=center_pos, pos=0, offset=offset)#, inpainted_images_gt=inpainted_images_gt, predicted_depths_gt=predicted_depths_gt, inpaint_masks_input_gt=inpaint_masks_gt)  #, inpainted_images=inpainted_images, predicted_depths=predicted_depths, inpaint_masks_input=inpaint_masks
        pipeline.clean_mesh()

        depth_mse = compute_depth_mse(pipeline, init_dataset, args, H=480, W=640)
        f_decision.write(f"Depth MSE {i}: {depth_mse}\n")
        if depth_mse < min_depth_mse or min_depth_mse == -1:
            min_depth_mse = depth_mse
            decision = i
            best_vertices, best_colors, best_faces = pipeline.vertices, pipeline.colors, pipeline.faces
        
    
    pipeline.vertices, pipeline.colors, pipeline.faces = best_vertices, best_colors, best_faces
    f_decision.write(f"\nDecision: {decision}\n")
    f_decision.close()
    
    del pano_pipeline

    return offset

def get_average_pos(args, init_dataset):
    # https://en.wikipedia.org/wiki/Camera_matrix#The_camera_position
    
    pos = torch.zeros((len(init_dataset), 3, 1), device=args.device)
    
    for i, data in enumerate(init_dataset):
        w2c = torch.tensor(data['w2c'], device=args.device)
        r = w2c[:3, :3]
        t = w2c[:3, 3:4]
        r_inv = torch.linalg.inv(r)
        pos[i, ...] = - r_inv @ t
    
    return torch.mean(pos, dim=0)

def get_random_pos_from_input_pos(args, init_dataset, num_sample=1):
    '''
    return a list of unique random positions from input data
    '''

    pos_indices = []
    random_pos = []
    
    for i in range(num_sample):
        pos_index = int(torch.randint(0, len(init_dataset), (1,)).cpu())
        if i < num_sample - 1:
            while pos_index in pos_indices:
                pos_index = int(torch.randint(0, len(init_dataset), (1,)).cpu())
        pos_indices.append(pos_index)

        w2c = torch.tensor(init_dataset[pos_index]['w2c'], device=args.device)
        r = w2c[:3, :3]
        t = w2c[:3, 3:4]
        r_inv = torch.linalg.inv(r)
        random_pos.append(- r_inv @ t)

    return random_pos

def get_center_pos_list(args, init_dataset):
    if args.panorama_active_sampling_always_at_center:
        center_pos_list = [get_average_pos(args, init_dataset)] * args.num_panorama_active_sampling
    
    else:
        num_sample = args.num_panorama_active_sampling - 1*args.panorama_active_sampling_must_include_center
        center_pos_list = get_random_pos_from_input_pos(args, init_dataset, num_sample)
        if args.panorama_active_sampling_must_include_center:
            center_pos_list.append(get_average_pos(args, init_dataset))
    
    return center_pos_list

def load_text2room_config(pipeline: GenRCPipeline):
    pipeline.args.inpaint_panorama_first = False
    pipeline.args.complete_mesh = False
    pipeline.args.guidance_scale = 7.5
    pipeline.args.negative_prompt = "blurry, bad art, blurred, text, watermark, plant, nature"
    pipeline.args.guidance_scale = 7.5
    pipeline.args.surface_normal_threshold = 0.1
    pipeline.args.edge_threshold = 0.1
    pipeline.args.min_triangles_connected = 15000
    pipeline.args.iron_depth_iters = 20
    

def inpaint_with_scannet_poses(pipeline: GenRCPipeline, init_dataset, offset=0, fov=None, shuffle=False):
    
    # generate poses using unsampled poses in the scannet scene
    w2cs = init_dataset.get_all_w2cs(exclude_init_frames=False)
    if shuffle:
        import random
        random.Random(2).shuffle(w2cs)
    offset = inpaint_with_poses(pipeline, w2cs, offset, fov)

    return offset  

def render_with_scannet_poses(pipeline: GenRCPipeline, dataset, save_path, fov=None, n_iterp=5, fps=10, depth_minmax=None):
    w2cs = torch.tensor(dataset.get_all_w2cs(), device=pipeline.args.device)
    rendered_image_pils, rendered_depth_pils, inpaint_mask_pils, rendered_rgbd_pils = render_with_poses(pipeline, w2cs, fov, n_iterp, depth_minmax=depth_minmax)
    img2video(rendered_image_pils, os.path.join(save_path, "rgb.mp4"), fps=fps)
    img2video(rendered_depth_pils, os.path.join(save_path, "depth.mp4"), fps=fps)
    img2video(inpaint_mask_pils, os.path.join(save_path, "mask.mp4"), fps=fps)
    img2video(rendered_rgbd_pils, os.path.join(save_path, "rgbd.mp4"), fps=fps)

def render_with_center_360_poses(pipeline: GenRCPipeline, dataset, save_path, fov=None, n_iterp=0, n_degrees=30, fps=20, depth_minmax=None):
    center_pos = get_average_pos(pipeline.args, dataset)
    w2cs = []
    w2cs.append(torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype = torch.float32, device = pipeline.args.device))
    
    degrees_per_rotate = 2*np.pi / n_degrees
    rot_matrix = torch.tensor([[np.cos(degrees_per_rotate), 0, np.sin(degrees_per_rotate), 0], [0, 1, 0, 0], [-np.sin(degrees_per_rotate), 0, np.cos(degrees_per_rotate), 0], [0, 0, 0, 1]], dtype = torch.float32, device = pipeline.args.device)
    for _ in range(n_degrees+1):
        w2cs.append(w2cs[-1]@rot_matrix)
    
    # offset for rotation
    rot_matrix = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype = torch.float32, device = pipeline.args.device)
    w2cs = [i@rot_matrix for i in w2cs]

    # add average translation of input poses
    w2cs = [torch.cat((i[:, :3], i[:, :3] @ (center_pos*-1)), dim=1) for i in w2cs]
    
    rendered_image_pils, rendered_depth_pils, inpaint_mask_pils, rendered_rgbd_pils = render_with_poses(pipeline, w2cs, fov, n_iterp, rotate=True, depth_minmax=depth_minmax)
    img2video(rendered_image_pils, os.path.join(save_path, "rgb_360.mp4"), fps=fps)
    img2video(rendered_depth_pils, os.path.join(save_path, "depth_360.mp4"), fps=fps)
    img2video(inpaint_mask_pils, os.path.join(save_path, "mask_360.mp4"), fps=fps)
    img2video(rendered_rgbd_pils, os.path.join(save_path, "rgbd_360.mp4"), fps=fps)

def img2video(pils, video_path, fps=10):

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    size = np.array(pils[0]).shape[:2]

    videoWrite = cv2.VideoWriter(video_path, fourcc, fps, (size[1],size[0])) 

    for pil in pils:
        # BGR -> RGB
        img = np.array(pil)[:,:,::-1]
        videoWrite.write(img)

    videoWrite.release()

def test_scannet_rgbd2(pipeline: GenRCPipeline, test_dataset, args):

    blur_before_metric_kernel_sizes = [-1] + [int(k.strip(' ')) for k in args.blur_before_metric_kernel_size.split(',')]
    for blur_before_metric_kernel_size in blur_before_metric_kernel_sizes:
        compute_metrics(pipeline, test_dataset, args, blur_before_metric_kernel_size=blur_before_metric_kernel_size)


def compute_metrics(pipeline: GenRCPipeline, test_dataset, args, outdir_name="test", H=180, W=240, blur_before_metric_kernel_size=-1, downsample=3):

    old_fov = pipeline.args.fov
    old_H = pipeline.H
    old_W = pipeline.W
    pipeline.args.fov = 58.480973919436906
    assert H%downsample==0 and W%downsample==0
    pipeline.H = H*downsample
    pipeline.W = W*downsample

    if downsample >= 1:
        save_path = os.path.join(args.out_path, outdir_name+f"_down{downsample}")
    else:
        save_path = os.path.join(args.out_path, outdir_name)

    if blur_before_metric_kernel_size > 0:
        save_path += f"_blur_k{blur_before_metric_kernel_size}"

    print(save_path)

    testing_log_path = os.path.join(save_path, 'log.txt')
    testing_metrics_path = os.path.join(save_path, 'metrics.txt')

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "depth"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "rendered_mask"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "depth_mask"), exist_ok=True)
    f_log = open(testing_log_path, 'w')
    f_metrics = open(testing_metrics_path, 'w')

    rgb_psnrs = []
    rgb_ssims = []
    depth_mses = []

    gt_images = []
    rendered_images = []

    for i, data in enumerate(test_dataset):
        w2c = torch.tensor(data['w2c'], device=args.device)
        intrinsic = torch.tensor(data['K'], device=args.device)
        intrinsic[:2] *= 0.375 * downsample

        gt_image = data['image']
        gt_depth = data['depth']
        gt_depth_mask = gt_depth == 0

        rendered_image, rendered_depth, rendered_mask = pipeline.render_with_pose(intrinsic, w2c)

        # inpaint rendered images (use the tranditional method to inpaint pixels without rendered values)
        rendered_image = cv2.inpaint(rendered_image, rendered_mask.astype('uint8'), 3, cv2.INPAINT_TELEA)
        # print(rendered_depth.max(), rendered_depth.min())
        inpainted_rendered_depth = cv2.inpaint(rendered_depth*1000, rendered_mask.astype('uint8'), 3, cv2.INPAINT_TELEA)/1000
        rendered_depth[rendered_mask] = inpainted_rendered_depth[rendered_mask]

    
        if downsample >= 1:
            rendered_image = cv2.resize(rendered_image, (W, H), interpolation = cv2.INTER_AREA)
            rendered_depth = cv2.resize(rendered_depth, (W, H), interpolation = cv2.INTER_AREA)

        if blur_before_metric_kernel_size > 0:
            k = blur_before_metric_kernel_size
            rendered_image = cv2.GaussianBlur(rendered_image, (k,k), 0)
            rendered_depth = cv2.GaussianBlur(rendered_depth, (k,k), 0)


        # save rendered images
        rendered_rgb_pil = Image.fromarray(rendered_image.astype(np.uint8))
        gt_rgb_pil = Image.fromarray(gt_image.astype(np.uint8))
        mixed_rgb_pil = Image.fromarray((rendered_image*0.5+gt_image*0.5).astype(np.uint8))
        rendered_depth_pil = Image.fromarray(visualize_depth_numpy(rendered_depth)[0])
        gt_depth_pil = Image.fromarray(visualize_depth_numpy(gt_depth)[0])
        save_image(mixed_rgb_pil, f"mixed_image", i, os.path.join(save_path, "rgb"))
        save_image_pair(gt_rgb_pil, rendered_rgb_pil, f"gt_rendered_image", i, os.path.join(save_path, "rgb"))
        save_image_pair(gt_depth_pil, rendered_depth_pil, f"gt_rendered_depth", i, os.path.join(save_path, "depth"))
        # normalize images
        gt_image = gt_image.astype('float64')
        rendered_image = rendered_image.astype('float64')

        # caculate overall errors
        # rendered_mask = rendered_mask
        depth_mask = gt_depth_mask #| rendered_mask

        rgb_psnr = calculate_psnr(gt_image, rendered_image) #, border=2)
        rgb_ssim = calculate_ssim(gt_image, rendered_image) #, border=2)
        depth_mse = calculate_masked_mse(gt_depth, rendered_depth, depth_mask)

        rgb_psnrs.append(rgb_psnr)
        rgb_ssims.append(rgb_ssim)
        depth_mses.append(depth_mse)

        save_image(Image.fromarray(rendered_mask), f"rendered_mask", i, os.path.join(save_path, "rendered_mask"))
        save_image(Image.fromarray(depth_mask), f"depth_mask", i, os.path.join(save_path, "depth_mask"))

        f_log.write(f'id:{i}, rgb_psnr: {rgb_psnr}, rgb_ssim: {rgb_ssim}, depth_mse: {depth_mse}\n')

        # save images for other metrics
        gt_images.append(gt_image)
        rendered_images.append(rendered_image)

    input_images = test_dataset.get_train_frames()

    avg_rgb_lpips = calculate_lpips(gt_images, rendered_images)
    avg_rgb_is_mean, avg_rgb_is_std = calculate_is(rendered_images)
    avg_rgb_fid_with_gt = calculate_fid(gt_images, rendered_images)
    avg_rgb_fid_with_input = calculate_fid(input_images, rendered_images)
    avg_rgb_cs_with_gt = calculate_cs(gt_images, rendered_images)
    avg_rgb_cs_with_input = calculate_cs(input_images, rendered_images)

    avg_rgb_psnr = np.stack(rgb_psnrs).mean()
    avg_rgb_ssim = np.stack(rgb_ssims).mean()
    avg_depth_mse = np.stack(depth_mses).mean()

    f_metrics.write(f'(overall) avg_rgb_psnr: {avg_rgb_psnr}, avg_rgb_ssim: {avg_rgb_ssim}, avg_depth_mse: {avg_depth_mse}\n')
    f_metrics.write(f'(overall) avg_rgb_lpips: {avg_rgb_lpips}, avg_rgb_is_mean: {avg_rgb_is_mean}, avg_rgb_is_std: {avg_rgb_is_std}, avg_rgb_fid_with_gt: {avg_rgb_fid_with_gt}, avg_rgb_fid_with_input: {avg_rgb_fid_with_input}, avg_rgb_cs_with_gt: {avg_rgb_cs_with_gt}, avg_rgb_cs_with_input: {avg_rgb_cs_with_input}\n')

    f_log.close()
    f_metrics.close()

    pipeline.args.fov = old_fov
    pipeline.H = old_H
    pipeline.W = old_W

    return avg_rgb_psnr, avg_rgb_ssim, avg_depth_mse
    
def compute_depth_mse(pipeline: GenRCPipeline, test_dataset, args, outdir_name="test", H=180, W=240, blur_before_metric_kernel_size=-1):

    old_fov = pipeline.args.fov
    old_H = pipeline.H
    old_W = pipeline.W
    pipeline.args.fov = 58.480973919436906
    pipeline.H = H
    pipeline.W = W

    depth_mses = []

    for i, data in enumerate(test_dataset):
        w2c = torch.tensor(data['w2c'], device=args.device)
        intrinsic = torch.tensor(data['K'], device=args.device)
        intrinsic[:2] *= H / 480

        gt_depth = data['depth']
        gt_depth_mask = gt_depth == 0

        rendered_image, rendered_depth, rendered_mask = pipeline.render_with_pose(intrinsic, w2c)

        inpainted_rendered_depth = cv2.inpaint(rendered_depth*1000, rendered_mask.astype('uint8'), 3, cv2.INPAINT_TELEA)/1000
        rendered_depth[rendered_mask] = inpainted_rendered_depth[rendered_mask]

        if blur_before_metric_kernel_size > 0:
            k = blur_before_metric_kernel_size
            rendered_depth = cv2.GaussianBlur(rendered_depth, (k,k), 0)

        # caculate overall errors
        # rendered_mask = rendered_mask
        depth_mask = gt_depth_mask #| rendered_mask

        depth_mse = calculate_masked_mse(gt_depth, rendered_depth, depth_mask)

        depth_mses.append(depth_mse)


    avg_depth_mse = np.stack(depth_mses).mean()

    pipeline.args.fov = old_fov
    pipeline.H = old_H
    pipeline.W = old_W

    return avg_depth_mse


# --------------------------------------------
# Chamfer Distance
# --------------------------------------------

def calculate_chamfer_distance(pipeline1, gt_pipeline, num_sample_points = 10000, single_directional=True, device="cuda"):
    '''
    Calculate Chamfer Distance
    '''
    texture = TexturesVertex(verts_features=[pipeline1.colors.T])
    mesh1 = Meshes(verts=[pipeline1.vertices.T], faces=[pipeline1.faces.T], textures=texture)
    mesh1_sampled = sample_points_from_meshes(mesh1, num_sample_points)

    texture = TexturesVertex(verts_features=[gt_pipeline.colors.T])
    gt_mesh = Meshes(verts=[gt_pipeline.vertices.T], faces=[gt_pipeline.faces.T], textures=texture)
    gt_mesh_sampled = sample_points_from_meshes(gt_mesh, num_sample_points)

    chamfer_dist, _ = chamfer_distance(gt_mesh_sampled, mesh1_sampled, single_directional=single_directional)

    return chamfer_dist


# --------------------------------------------
# Completeness
# --------------------------------------------

def calculate_completeness(pipeline1, gt_pipeline, threshold=0.1, num_sample_points = 10000, device="cuda"):
    '''
    Calculate Completeness, aka percentage of vertices in gt mesh
    that have any vertices within radius of threshold in generated mesh
    '''

    texture = TexturesVertex(verts_features=[pipeline1.colors.T])
    mesh1 = Meshes(verts=[pipeline1.vertices.T], faces=[pipeline1.faces.T], textures=texture)
    mesh1_sampled = sample_points_from_meshes(mesh1, num_sample_points)

    texture = TexturesVertex(verts_features=[gt_pipeline.colors.T])
    gt_mesh = Meshes(verts=[gt_pipeline.vertices.T], faces=[gt_pipeline.faces.T], textures=texture)
    gt_mesh_sampled = sample_points_from_meshes(gt_mesh, num_sample_points)

    x_nn = knn_points(gt_mesh_sampled, mesh1_sampled, K=1)
    dist_x = torch.sqrt(x_nn.dists[0, :, 0])
    completeness = dist_x[dist_x <= threshold].shape[0] / num_sample_points
    
    return completeness
