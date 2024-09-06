import os
from model.genrc_pipeline import GenRCPipeline
from model.utils.opt import get_default_parser
from evaluate.utils import test_scannet_rgbd2, get_average_pos, init_scannet, calculate_chamfer_distance, calculate_completeness
from glob import glob
from tqdm import tqdm

import torch
import numpy as np
import random

from dataset.scannet_rgbd2 import ScanNetRGBD2

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def main(args):

    args.save_files = False
    args.use_only_gt_depth = True
    args.replace_overlapping = False
    args.simplify_mesh_voxel_size = 0.02
    args.outlier_point_radius = -1

    chamfer_single_directional = True
    num_sample_points = 10000           # as mentioned in RGBD2 paper

    data_path = args.data_path
    
    # path to Our mesh
    load_path = os.path.join(args.out_path, args.exp_name)
    
    # path to ground truth mesh
    gt_path = load_path
    gt_scene_names = os.listdir(gt_path)
    gt_scene_names.sort()
    
    with open(os.path.join(data_path, "test_scenes.txt"),'r') as fp:
        test_scenes = fp.readlines()
        test_scenes = [x.strip('\n') for x in test_scenes]

    sample_percents = [5, 10, 20, 50]
    blur_kernel_sizes = [-1, 5]


    for scene_name, gt_scene_name in tqdm(zip(test_scenes, gt_scene_names), total = len(test_scenes)):
        for percent in sample_percents:
            
            # load pipeline
            pipeline = GenRCPipeline(args, setup_models=False, no_save_dir=True)

            mesh_path = os.path.join(load_path, scene_name, str(percent), "*", "fused_mesh", "after_sample_given_poses.ply")
            mesh_path = glob(mesh_path)[0]
            pipeline.load_mesh(mesh_path)

            # load gt pipeline
            gt_pipeline = GenRCPipeline(args, setup_models=False, no_save_dir=True)

            #mesh_path = os.path.join(gt_path, scene_name, "100", "*", "fused_mesh", "initialization.ply")   # ours
            #mesh_path = os.path.join(gt_path, gt_scene_name, "100%", f"{gt_scene_name}.ply")                # rgbd2
            
            mesh_path = os.path.join(load_path, scene_name, "100", "*", "fused_mesh", "unprojected_gt_no_inpaint_depth.ply")
            mesh_path = glob(mesh_path)

            if len(mesh_path) == 0:
                init_dataset = ScanNetRGBD2(data_path, scene_name, init_frames=1.0, z_offset=args.z_offset, split='train', resize=True)
                init_scannet(gt_pipeline, init_dataset, args, 0)
                gt_pipeline.clean_mesh()

                # save mesh
                mesh_path = os.path.join(load_path, scene_name, str(percent), "*")
                mesh_path = glob(mesh_path)[0]
                gt_pipeline.args.fused_mesh_path = os.path.join(load_path, scene_name, "100", os.path.basename(mesh_path), "fused_mesh")
                os.makedirs(gt_pipeline.args.fused_mesh_path, exist_ok=True)
                gt_pipeline.save_mesh(f"unprojected_gt_no_inpaint_depth.ply")
            else:
                mesh_path = mesh_path[0]
                gt_pipeline.load_mesh(mesh_path)
                gt_pipeline.clean_mesh()

            # calculate Chamfer Distance and Completeness
            chamfer_dist = calculate_chamfer_distance(pipeline, gt_pipeline, num_sample_points, chamfer_single_directional)
            completeness = calculate_completeness(pipeline, gt_pipeline, 0.1, num_sample_points)

            # save file
            for blur_kernel_size in blur_kernel_sizes:
                test_dir_name = ("test_down3"+f"_blur_k{blur_kernel_size}") if blur_kernel_size != -1 else "test_down3"

                testing_dir_path = os.path.join(load_path, scene_name, str(percent), "*", test_dir_name)
                testing_dir_path = glob(testing_dir_path)[0]
                
                testing_chamfer_path = os.path.join(testing_dir_path, 'chamfer_distance.txt')
                f_chamfer = open(testing_chamfer_path, 'w')
                f_chamfer.write(f"{chamfer_dist}")
                f_chamfer.close()

                testing_completeness_path = os.path.join(testing_dir_path, 'completeness.txt')
                f_completeness = open(testing_completeness_path, 'w')
                f_completeness.write(f"{completeness}")
                f_completeness.close()

            del pipeline, gt_pipeline
            # reset gpu memory
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    main(args)
