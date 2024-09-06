import os
from model.genrc_pipeline import GenRCPipeline
from model.utils.opt import get_default_parser
from model.utils.utils import setup_seed, save_poisson_mesh
from evaluate.utils import init_scannet, inpaint_with_scannet_poses, test_scannet_rgbd2, compute_depth_mse, inpaint_panorama, get_average_pos, get_center_pos_list, load_text2room_config
from dataset.scannet_rgbd2 import ScanNetRGBD2
from dataset.ARKit import ARKit_2
import torch
import copy

@torch.no_grad()
def main(args):
   
    data_path = args.data_path
    textual_inversion_token_path = 'input/ScanNetV2_test'
    outdir = os.path.join(args.out_path, args.exp_name)

    
    with open(os.path.join(data_path, "test_scenes.txt"),'r') as fp:
        test_scenes = fp.readlines()
        test_scenes = [x.strip('\n') for x in test_scenes]

    
    # edit here for testing configs
    sample_percents = [5, 10, 20, 50]
    #test_scenes.reverse()
    #test_scenes = test_scenes[:1]

    for percent in sample_percents:
        for scene_name in test_scenes:
            
            scene_outdir = os.path.join(outdir, scene_name)
            sampled_scene_outdir = os.path.join(scene_outdir, str(percent))
            
            if not os.path.exists(sampled_scene_outdir):
                os.makedirs(sampled_scene_outdir)

            # setup configs
            args.view_percent = percent
            args.scene_id = scene_name
            args.out_path = sampled_scene_outdir

            # comment this out to diable textual inversion
            args.textual_inversion_path = os.path.join(textual_inversion_token_path, scene_name, 
                                                    'textual_inversion', str(percent), 
                                                    'learned_embeds-steps-3000.bin')

            # setup the random seed
            setup_seed(args.seed)

            # load pipeline
            pipeline = GenRCPipeline(args)

            # reset the offset
            offset = 0

            # setup init dataset
            init_dataset = ScanNetRGBD2(data_path, args.scene_id, init_frames=args.view_percent*0.01, z_offset=args.z_offset, split='train', resize=True)
            offset = init_scannet(pipeline, init_dataset, args, offset)
            pipeline.clean_mesh()
            pipeline.save_mesh(f"initialization.ply")

            # complete the room using the baseline method, Text2room.
            if args.method == "T2R+RGBD":
                load_text2room_config(pipeline)

            # inpaint panorama
            if args.inpaint_panorama_first:
                inpaint_panorama(pipeline, init_dataset, args, offset)
                pipeline.save_mesh("after_panorama.ply")
                
            # sample novel views to patch up holes (default disabled)
            if args.complete_mesh:
                offset = pipeline.complete_mesh(offset)
                pipeline.clean_mesh()
                pipeline.save_mesh("after_completion.ply")

            # inpaint with the scannet testing poses
            offset = inpaint_with_scannet_poses(pipeline, init_dataset, offset)
            pipeline.clean_mesh()
            after_sample_given_poses_path = pipeline.save_mesh("after_sample_given_poses.ply")

            # poisson recontruciton (default disabled)
            if args.poisson_reconstruct_mesh:
                mesh_poisson_path = save_poisson_mesh(after_sample_given_poses_path, depth=args.poisson_depth, max_faces=args.max_faces_for_poisson)
            
            # run evaluation
            pipeline.remove_models()
            testing_dataset = ScanNetRGBD2(data_path, args.scene_id, init_frames=args.view_percent*0.01, z_offset=args.z_offset, split='test', resize=False)
            test_scannet_rgbd2(pipeline, testing_dataset, args)

            del pipeline
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    main(args)
