import torch
import cv2
import os
import json
import numpy as np
import time
import pymeshlab
import imageio
import random
import open3d as o3d

from PIL import Image

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionInpaintPipeline
from ..inpaint_one_step.inpaint_one_step import SDInpaintOneStepPipeline

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_TURBO):
    """
    depth: (H, W)
    """
    x = np.nan_to_num(depth)  # change nan to 0
    mask_nan = (x == 0) # assume that the pixels without depth are 0s
    if minmax is None:
        if (x > 0).any():
            mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
            ma = np.max(x)
        else:
            mi = 0.0
            ma = 0.0
    else:
        mi, ma = minmax
        x = np.clip(x, mi, ma)

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    x_[mask_nan] *= 0 # assign 0 to pixels without depth
    return x_, [mi, ma]


def load_sd_inpaint_one_step(args):
    model_path = os.path.join(args.models_path, "stable-diffusion-2-inpainting")
    if not os.path.exists(model_path):
        model_path = "stabilityai/stable-diffusion-2-inpainting"
    pipe = SDInpaintOneStepPipeline.from_pretrained(model_path, torch_dtype=torch.float16, cache_dir="pretrained_weight_cache/").to(args.device)

    if args.textual_inversion_path is not None:
        pipe.load_textual_inversion(args.textual_inversion_path)

    pipe.set_progress_bar_config(**{
        "leave": False,
        "desc": "Generating Next Image"
    })

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    #pipe.enable_sequential_cpu_offload()    #  reduce memory

    return pipe

def pil_to_torch(img, device, normalize=True):
    img = torch.tensor(np.array(img), device=device).permute(2, 0, 1)
    if normalize:
        img = img / 255.0
    return img


def generate_first_image(args):
    model_path = os.path.join(args.models_path, "stable-diffusion-2-1")
    if not os.path.exists(model_path):
        model_path = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)

    pipe.set_progress_bar_config(**{
        "leave": False,
        "desc": "Generating Start Image"
    })

    return pipe(args.prompt).images[0]


def save_image(image, prefix, idx, outdir):
    filename = f"{prefix}_{idx:04}"
    ext = "png"
    file_with_ext = f"{filename}.{ext}"
    file_out = os.path.join(outdir, file_with_ext)
    image.save(file_out)
    return file_with_ext

def save_image_pair(image_first, image_second, prefix, idx, outdir):
    assert image_first.height == image_second.height
    filename = f"{prefix}_{idx:04}"
    ext = "png"
    file_with_ext = f"{filename}.{ext}"
    file_out = os.path.join(outdir, file_with_ext)
    dst = Image.new('RGB', (image_first.width + image_second.width, image_first.height))
    dst.paste(image_first, (0, 0))
    dst.paste(image_second, (image_first.width, 0))
    dst.save(file_out)
    return file_with_ext

def save_rgbd(image, depth, prefix, idx, outdir):
    filename = f"{prefix}_{idx:04}"
    ext = "png"
    file_with_ext = f"{filename}.{ext}"
    file_out = os.path.join(outdir, file_with_ext)
    dst = Image.new('RGB', (image.width + depth.width, image.height))
    dst.paste(image, (0, 0))
    dst.paste(depth, (image.width, 0))
    dst.save(file_out)
    return file_with_ext

def get_rgbd_image(image, depth):
    rgbd = Image.fromarray(np.concatenate((np.array(image),np.array(depth)), axis=1))
    return rgbd 

def save_settings(args):
    with open(os.path.join(args.out_path, "settings.json"), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)


def save_animation(image_folder_path, prefix=""):
    gif_name = os.path.join(image_folder_path, prefix + 'animation.gif')
    images = [os.path.join(image_folder_path, img) for img in sorted(os.listdir(image_folder_path), key=lambda x: int(x.split(".")[0].split("_")[-1])) if "rgb" in img]

    with imageio.get_writer(gif_name, mode='I', duration=0.2) as writer:
        for filename in images:
            image = imageio.v3.imread(filename)
            writer.append_data(image)


def save_poisson_mesh(mesh_file_path, depth=12, max_faces=10_000_000):
    # load mesh
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_file_path)
    print("loaded", mesh_file_path)

    # compute normals
    start = time.time()
    ms.compute_normal_for_point_clouds()
    print("computed normals")

    # run poisson
    ms.generate_surface_reconstruction_screened_poisson(depth=depth)
    end_poisson = time.time()
    print(f"finish poisson in {end_poisson - start} seconds")

    # save output
    parts = mesh_file_path.split(".")
    out_file_path = ".".join(parts[:-1])
    suffix = parts[-1]
    out_file_path_poisson = f"{out_file_path}_poisson_meshlab_depth_{depth}.{suffix}"
    ms.save_current_mesh(out_file_path_poisson)
    print("saved poisson mesh", out_file_path_poisson)

    # quadric edge collapse to max faces
    start_quadric = time.time()
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=max_faces)
    end_quadric = time.time()
    print(f"finish quadric decimation in {end_quadric - start_quadric} seconds")

    # save output
    out_file_path_quadric = f"{out_file_path}_poisson_meshlab_depth_{depth}_quadric_{max_faces}.{suffix}"
    ms.save_current_mesh(out_file_path_quadric)
    print("saved quadric decimated mesh", out_file_path_quadric)

    return out_file_path_poisson

def load_poisson_mesh(mesh_file_path):
    # load mesh
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_file_path)
    print("loaded", mesh_file_path)

def get_cubemap_w2c(center_pos=None):
    '''
    returns a list of six 90-fov w2c matrix for cubemaps centered at center_pos
    0-front 1-right 2-back 3-left, 4-top, 5-bottom (aka counter-clockwise)
    '''
    if center_pos is not None:
        device = center_pos.device
    else:
        device = "cpu"
    
    cubemap_world_to_cams = []
    
    # four sides
    cubemap_world_to_cams.append(torch.tensor([[1, 0, 0, 0], [0, 1,0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=device, dtype=torch.float32))
    
    rot_matrix = torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], device=device, dtype=torch.float32)
    for i in range(3):
        cubemap_world_to_cams.append(torch.matmul(cubemap_world_to_cams[-1],rot_matrix))
        
    # up
    rot_matrix = torch.tensor([[1, 0, 0, 0], [0, 0,-1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], device=device, dtype=torch.float32)
    cubemap_world_to_cams.append(torch.matmul(cubemap_world_to_cams[0],rot_matrix))
    
    # down
    rot_matrix = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], device=device, dtype=torch.float32)
    cubemap_world_to_cams.append(torch.matmul(cubemap_world_to_cams[0],rot_matrix))

    # offset for rotation
    rot_matrix = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], device=device, dtype=torch.float32)
    cubemap_world_to_cams = [torch.matmul(i,rot_matrix) for i in cubemap_world_to_cams]

    # add average translation of input poses
    if center_pos is not None:
        cubemap_world_to_cams = [torch.cat((i[:, :3], i[:, :3] @ (center_pos*-1)), dim=1) for i in cubemap_world_to_cams]

    return cubemap_world_to_cams

def depth_to_distance_ratio( H, W, intrinsics, device=None):    
    '''
    Ratio converting from depthmap aka "distance between pixels and image plane"
    to "distance between pixels and camera origin"
    '''
    if device is None:
        device = intrinsics.device

    # Get the intrinsic parameters
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Create the pixel grid
    rows = H
    cols = W
    u = torch.arange(0, cols, dtype=torch.float32, device=device).repeat(rows, 1)
    v = torch.arange(0, rows, dtype=torch.float32,device=device).repeat(cols, 1).t()

    # Calculate the distance from camera origin to each pixel
    x = (u - cx) / fx
    y = (v - cy) / fy

    distance = torch.sqrt(x ** 2 + y ** 2 + 1)

    return distance

def erode_and_dilate_mask(mask):
    """
    mask: torch tensor of shape (3, H, W)
    """
    
    # mask dilation
    m = mask[0, ...].cpu().numpy().astype(np.uint8)
    m[m!=0] = -1
    
    # remove small seams from mask
    m2 = cv2.erode(m, (3, 3), iterations=1)
    for _ in range(2):
        m2 = cv2.GaussianBlur(m2, (7, 7), 0)
    m2[m2!=0] = -1
    
    mask = torch.cat([torch.tensor(m2, dtype=mask.dtype, device=mask.device).unsqueeze(0)] * 3, dim=0)
    mask[mask != 0] = 1

    return mask