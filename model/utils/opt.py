import argparse


def get_default_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # GENERAL CONFIG
    group = parser.add_argument_group("general")
    group.add_argument('--method', type=str, default="GenRC")
    group.add_argument('--device', required=False, default="cuda")
    group.add_argument('--seed', required=False, type=int, default=0)
    group.add_argument('--exp_name', type=str, default='ScanNetV2_exp')
    group.add_argument('--trajectory_file', type=str, default='model/trajectories/examples/living_room.json')
    group.add_argument('--models_path', required=False, default="checkpoints_link")
    group.add_argument('--out_path', required=False, default="output")
    group.add_argument('--input_image_path', required=False, type=str, default=None)
    group.add_argument('--n_images', required=False, default=10, type=int)
    group.add_argument('--save_scene_every_nth', required=False, default=-1, type=int)
    group.add_argument('--save_files', required=False, default=True)

    # SCANNET CONFIG
    group = parser.add_argument_group("scannet")
    group.add_argument('--data_path', required=False, default='input/ScanNetV2', type=str)
    group.add_argument('--scene_id', required=False, default='0000_00', type=str)
    group.add_argument('--view_percent', required=False, default=5, type=float)
    group.add_argument('--axis_aligment', required=False, default=False, action="store_true", help="align initial mesh to xy-plane like text2room. (If use trajectory files, it should be true.)")
    group.add_argument('--z_offset', required=False, default=1.2, type=float, help="shift initial mesh along z-axis. (it may influence rendered and inpainting results)")
    group.add_argument('--replace_overlapping', required=False, default=True, help="remove overlapping pixels of the given images")
    group.add_argument('--use_only_gt_depth', required=False, default=False, action="store_true", help="don't inpaint lost values of input gt depths")
    group.add_argument('--mask_inpainted_depth_edges', required=False, default=False, action="store_true", help="do edge masking at inapinting results (i.e. don't project edges to the mesh)")
    group.add_argument('--inpaint_initialized_images_with_fov', required=False, default=None, type=float, help="do inpainting with a bigger fov for each given input image")


    # DEPTH CONFIG
    group = parser.add_argument_group("depth")
    group.add_argument('--iron_depth_type', required=False, default="scannet", type=str, help="model type of iron_depth")
    group.add_argument('--iron_depth_iters', required=False, default=100, type=int, help="amount of refinement iterations per prediction")
    group.add_argument('--gaussian_blur_before_predict_depth', required=False, default=False, action="store_true", help="do gaussian blur on the input image before predicting depth")

    # PROJECTION CONFIG
    group = parser.add_argument_group("projection")
    group.add_argument('--fov', required=False, default=58.480973919436906, type=float, help="FOV in degrees for all images")
    group.add_argument('--blur_radius', required=False, default=0, type=float, help="if render_mesh: blur radius from pytorch3d rasterization")

    # STABLE DIFFUSION CONFIG
    group = parser.add_argument_group("stable-diffusion")
    group.add_argument('--prompt', required=False, default="")
    group.add_argument('--negative_prompt', required=False, default="human, animals, plants, trees, green nature, watermarks, texts, letters, words, fonts, languages, alphabets, numbers, digits, typography, bad arts, distortions, blurry, messy, unstructured, discontinuous, inconsistent, truncated, clipped, complex, furniture, sundries, clothes, objects, small objects, box, groceries, litters, fragments, pieces, sections, technology, industry, machine, device, tools, pipes, wires")
    group.add_argument('--guidance_scale', required=False, default=5, type=float)
    group.add_argument('--num_inference_steps', required=False, default=50, type=int)
    group.add_argument('--num_inpainting_sampling', required=False, default=1, help="if >1, sample multiple inpainting results and pick the best one")
    group.add_argument('--pick_by_variance', required=False, default=False, type=bool, help="pick the best inpainting result by the depth variance (if true) instead of mean (if false)")
    group.add_argument('--blur_before_inpainting', required=False, default=False, action="store_true")

    # STABLE DIFFUSION CONFIG
    group = parser.add_argument_group("e-diffusion-and-panorama-depth-prediciton")
    group.add_argument('--inpaint_panorama_first', required=False, default=True, type=bool, help="whether to first inpaint panorama at room center")
    group.add_argument('--panorama_num_inference_steps', required=False, default=50, type=int)
    group.add_argument('--guidance_scale_panorama', required=False, default=5, type=float)
    group.add_argument('--panorama_iron_depth_iters', required=False, default=20, type=int, help="amount of refinement iterations per prediction")
    group.add_argument('--num_panorama_active_sampling', required=False, default=3, type=int, help="sample panorama at room center and pick the best one")
    group.add_argument('--panorama_active_sampling_must_include_center', required=False, default=True, type=bool, help="one of the sample must be at center")
    group.add_argument('--panorama_active_sampling_always_at_center', required=False, default=True, type=bool, help="always sample at room center")
    group.add_argument('--panorama_diffusion_hybrid_split', required=False, default=0.6, type=float, help="Default=0.6; setting as 1 would make it E-Diffuion only and setting as 0 would make it MultiDiffusion")
    group.add_argument('--panorama_stitching_alpha_blending_at_border_width', required=False, default=0, type=float, help="should be a value between 0 - 0.5.  Blend the border between perspective views when stitching panorama.")

    # INPAINTING MASK CONFIG
    group = parser.add_argument_group("inpaint")
    group.add_argument('--erode_iters', required=False, default=1, type=int, help="how often to erode the inpainting mask")
    group.add_argument('--dilate_iters', required=False, default=2, type=int, help="how often to dilate the inpainting mask")
    group.add_argument('--boundary_thresh', required=False, default=10, type=int, help="how many pixels from image boundaries may not be dilated")
    group.add_argument('--update_mask_after_improvement', required=False, default=True, help="after erosion/dilation/fill_contours -- directly update inpainting mask")
    group.add_argument('--mask_backward_facing_surface', required=False, default=False, action="store_true", help="mask the pixels of backward-facing surfaces to inpaint")

    # MESH UPDATE CONFIG
    group = parser.add_argument_group("fusion")
    group.add_argument('--edge_threshold', required=False, default=0.1, type=float, help="only save faces whose edges _all_ have a smaller l2 distance than this. Default: -1 (=skip)")
    group.add_argument('--surface_normal_threshold', required=False, default=0.01, type=float, help="only save faces whose normals _all_ have a bigger dot product to view direction than this. Default: -1 (:= do not apply threshold)")
    group.add_argument('--faces_per_pixel', required=False, default=8, type=int, help="how many faces per pixel to render (and accumulate)")
    group.add_argument('--replace_over_inpainted', required=False, default=False, action="store_true", help="remove existing faces at the inpainting mask. Note: if <update_mask_after_improvement> is not set, will update mask before performing this operation.")
    group.add_argument('--remove_faces_depth_threshold', required=False, default=0.1, type=float, help="during <replace_over_inpainted>: only remove faces that are within the threshold to rendered depth")
    group.add_argument('--clean_mesh_every_nth', required=False, default=20, type=int, help="run several cleaning steps on the mesh to remove noise, small connected components, etc.")
    group.add_argument('--min_triangles_connected', required=False, default=1000, type=int, help="during <clean_mesh_every_nth>: minimum number of connected triangles in a component to not be removed")
    group.add_argument('--outlier_point_radius', required=False, default=0.02, type=int, help="during <clean_mesh_every_nth>: point clouds who have small numbers of neighbors will be removed from the mesh")
    group.add_argument('--poisson_reconstruct_mesh', required=False, default=False, action="store_true", help="do poisson recontruction as the last step to beautify the mesh")
    group.add_argument('--poisson_depth', required=False, default=12, type=int, help="depth value to use for poisson surface reconstruction")
    group.add_argument('--max_faces_for_poisson', required=False, default=10_000_000, type=int, help="after poisson surface reconstruction: save another version of the mesh that has at most this many triangles")
    group.add_argument('--simplify_mesh_voxel_size', required=False, default=0.01, type=float, help="simplify mesh with voxel_size after updating the mesh. Don't do anything if -1")

    # completion parameters
    group = parser.add_argument_group("completion")
    group.add_argument('--complete_mesh', required=False, default=False, action="store_true")
    group.add_argument('--complete_mesh_iter', type=int, default=30, help="number of views to be inpainted")
    group.add_argument('--complete_mesh_num_sample_per_iter', type=int, default=200, help="select an optimal camera pose from these number of samples")
    group.add_argument("--complete_mesh_step_back_length", type=float, default=0.25, help="move the camera by this distance until criteria are not satisfied")
    group.add_argument('--completion_sample_resolution', type=int, default=64, help="reduce render resolution during sampling to speed up")
    group.add_argument("--completion_camera_elevation_angle_limit", type=float, default=15, help="elevation angle of sampled views must be within +-15 degrees")
    group.add_argument("--num_inpainting_sampling_completion", type=int, default=3, help="sample three inpaint results and pick the one with greatest avg depth")
    group.add_argument('--n_voxels', type=int, default=1000, help="how many voxels we use for this scene")
    group.add_argument('--n_dir', type=int, default=8, help="how many rotation directions we use for a camera inside one voxel")
    group.add_argument('--core_ratio_x', type=float, default=0.8, help="The ratio of regions we sample cameras compared to the bbox along x direction")
    group.add_argument("--core_ratio_y", type=float, default=0.8, help="The ratio of regions we sample cameras compared to the bbox along y direction")
    group.add_argument("--core_ratio_z", type=float, default=0.1, help="The ratio of regions we sample cameras compared to the bbox along y direction")
    group.add_argument('--minimum_completion_pixels', type=int, default=1000, help="we only inpaint images once there are over this many pixels to be inpainted")
    group.add_argument("--min_camera_distance_to_mesh", type=float, default=0.1, help="a sampled camera's position must be at least this far away from the mesh")
    group.add_argument("--min_depth_quantil_to_mesh", type=float, default=1.0, help="a sampled camera's observed image must have a 10% depth quantil of at least this depth")
    group.add_argument("--max_inpaint_ratio", type=float, default=0.5, help="inpaint ratio = inpaint area / image area")
    group.add_argument("--min_inpaint_ratio", type=float, default=0.01, help="inpaint ratio = inpaint area / image area")
    group.add_argument("--max_backface_ratio", type=float, default=0.01, help="must not contain more than this ratio of back faces")
    group.add_argument("--completion_dilate_iters", type=int, default=8, help="repeat mask dilation this many times during completion")

    # textual inversion parameters
    group = parser.add_argument_group("textual-inversion")
    group.add_argument('--textual_inversion_path', required=False, type=str, default=None)
    group.add_argument('--textual_inversion_token_name', type=str, default='<style>')

    # metric calculating parameters
    group = parser.add_argument_group("metric-calculating")
    group.add_argument('--blur_before_metric_kernel_size', required=False, type=str, default="5, 7, 11", help="Also computing metric results after blurring rendered results")
    group.add_argument('--chamfer_distance_gt_mesh_path', required=False, type=str, default="", help="Also computing metric results after blurring rendered results")

    return parser
