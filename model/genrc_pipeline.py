import torch
import math
import os
import cv2
from PIL import Image
import numpy as np
import json
from datetime import datetime
from tqdm.auto import tqdm
import trimesh
from scipy.spatial.transform import Rotation  

from model.mesh_fusion.util import (
    get_pinhole_intrinsics_from_fov,
    torch_to_trimesh
)

from model.mesh_fusion.render import (
    features_to_world_space_mesh,
    render_mesh,
    save_mesh,
    load_mesh,
    clean_mesh,
    edge_threshold_filter,
    simplify_mesh
)

from model.iron_depth.predict_depth import load_iron_depth_model, predict_iron_depth
from model.iron_depth.predict_depth_panorama import load_iron_depth_model_panorama, predict_iron_depth_panorama
from model.iron_depth.utils.utils import normal_to_rgb

from model.depth_alignment import depth_alignment

from model.trajectories import trajectory_util, pose_noise_util

from model.utils.utils import (
    visualize_depth_numpy,
    save_image,
    pil_to_torch,
    save_rgbd,
    get_rgbd_image,
    save_settings,
    save_animation,
    get_cubemap_w2c,
    depth_to_distance_ratio,
)

from equilib import cube2equi, equi2pers, equi2cube
from model.PerspectiveAndEquirectangular.lib import Equirec2Perspec as E2P
from model.PerspectiveAndEquirectangular.lib import multi_Perspec2Equirec as m_P2E
from torchvision import transforms

from model.diffusion.diffusion_pipeline import DiffusionPipeline

import time


import yaml

class GenRCPipeline(torch.nn.Module):
    def __init__(self, args, setup_models=True, H=512, W=512, no_save_dir=False):
        super().__init__()
        # setup (create out_dir, save args)
        self.args = args
        self.orig_n_images = self.args.n_images
        self.orig_surface_normal_threshold = self.args.surface_normal_threshold
        self.H = H
        self.W = W
        self.bbox = [torch.ones(3) * -1.0, torch.ones(3) * 1.0]  # initilize bounding box of meshs as [-1.0, -1.0, -1.0] -> [1.0, 1.0, 1.0]
        if not no_save_dir:
            self.setup_output_directories()

        assert H == 512 and W == 512, "stable_diffusion inpainting model can process only 512x512 images"

        if self.args.textual_inversion_path is not None:
            self.args.prompt = f"a simple and clean room in the style of {self.args.textual_inversion_token_name}"
        else:
            self.args.prompt = f"a simple and clean room"

        # load models if required
        if setup_models:
            self.setup_models()

        # initialize global point-cloud / mesh structures
        self.rendered_depth = torch.zeros((H, W), device=self.args.device)  # depth rendered from point cloud
        self.inpaint_mask = torch.ones((H, W), device=self.args.device, dtype=torch.bool)  # 1: no projected points (need to be inpainted) | 0: have projected points
        self.vertices = torch.empty((3, 0), device=args.device)
        self.colors = torch.empty((3, 0), device=args.device)
        self.faces = torch.empty((3, 0), device=args.device, dtype=torch.long)
        self.pix_to_face = None

        # initialize trajectory
        self.world_to_cam = torch.eye(4, dtype=torch.float32, device=self.args.device)
        self.K = get_pinhole_intrinsics_from_fov(H=self.H, W=self.W, fov_in_degrees=self.args.fov).to(self.world_to_cam)

    def init_with_posed_images(self, first_image_pils, poses, intrinsics, gt_depths=None, inpaint_masks=None, offset=0):
        
        # save start image if specified
        H_origin, W_origin = self.H, self.W
        
        for i, (first_image_pil, pose, intrinsic) in enumerate(tqdm(zip(first_image_pils, poses, intrinsics), total=len(first_image_pils), desc="Init with input images")):

            self.H, self.W = np.array(first_image_pil).shape[:2]
            self.K = intrinsic
            self.world_to_cam = pose
            self.rendered_depth = torch.zeros((self.H, self.W), device=self.args.device)  # depth rendered from point cloud
            self.inpaint_mask = inpaint_masks[i] if inpaint_masks is not None \
                else torch.ones((self.H, self.W), device=self.args.device, dtype=torch.bool)  # 1: no projected points (need to be inpainted) | 0: have projected points

            gt_depth = gt_depths[i] if gt_depths is not None else None
            
            self.setup_start_image(first_image_pil, offset=offset, gt_depth=gt_depth, use_opencv_camera_with_intrinsic=intrinsic)
            
            self.clean_mesh()

            offset += 1

        self.H, self.W = H_origin, W_origin
        self.K = get_pinhole_intrinsics_from_fov(H_origin, W_origin, self.args.fov).to(self.world_to_cam)
        self.world_to_cam = torch.eye(4, dtype=torch.float32, device=self.args.device)
        self.rendered_depth = torch.zeros((H_origin, W_origin), device=self.args.device)  # depth rendered from point cloud
        self.inpaint_mask = torch.ones((H_origin, W_origin), device=self.args.device, dtype=torch.bool)  # 1: no projected points (need to be inpainted) | 0: have projected points

        return offset
    
    def setup_start_image(self, first_image_pil, offset, gt_depth=None, use_opencv_camera_with_intrinsic=None):
        # predict depth, add 3D structure
        _, _, _, _, predicted_depth = self.add_next_image(inpainted_image_pils=[first_image_pil], pos=0, offset=offset, \
                                                          gt_depth=gt_depth, use_opencv_camera_with_intrinsic=use_opencv_camera_with_intrinsic)
        self.predicted_depth = predicted_depth

    def setup_models(self):
        # construct inpainting stable diffusion pipeline
        self.diffusion_pipe = DiffusionPipeline(self.args)

        # construct depth model
        self.iron_depth_n_net, self.iron_depth_model = load_iron_depth_model_panorama(self.args.iron_depth_type, self.args.iron_depth_iters, self.args.panorama_iron_depth_iters, self.args.models_path, self.args.device)

    def remove_models(self):
        self.diffusion_pipe = None
        self.iron_depth_model = None
        self.iron_depth_n_net = None
        torch.cuda.empty_cache()

    def setup_output_directories(self):
        now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%Ss.%fZ')
        self.args.out_path = os.path.join(self.args.out_path, now_str)
        os.makedirs(self.args.out_path, exist_ok=True)
        self.args.rgb_path = os.path.join(self.args.out_path, "rgb")
        self.args.rgbd_path = os.path.join(self.args.out_path, "rgbd")
        self.args.depth_path = os.path.join(self.args.out_path, "depth")
        self.args.rendered_rgb_path = os.path.join(self.args.out_path, "rendered_rgb")
        self.args.rendered_rgbd_path = os.path.join(self.args.out_path, "rendered_rgbd")
        self.args.rendered_depth_path = os.path.join(self.args.out_path, "rendered_depth")
        self.args.fused_mesh_path = os.path.join(self.args.out_path, "fused_mesh")
        self.args.mask_path = os.path.join(self.args.out_path, "mask")
        self.args.output_rendering_path = os.path.join(self.args.out_path, "output_rendering")
        self.args.output_depth_path = os.path.join(self.args.out_path, "output_depth")
        os.makedirs(self.args.rgb_path, exist_ok=True)
        os.makedirs(self.args.rgbd_path, exist_ok=True)
        os.makedirs(self.args.depth_path, exist_ok=True)
        os.makedirs(self.args.rendered_rgb_path, exist_ok=True)
        os.makedirs(self.args.rendered_rgbd_path, exist_ok=True)
        os.makedirs(self.args.rendered_depth_path, exist_ok=True)
        os.makedirs(self.args.fused_mesh_path, exist_ok=True)
        os.makedirs(self.args.mask_path, exist_ok=True)
        os.makedirs(self.args.output_rendering_path, exist_ok=True)
        os.makedirs(self.args.output_depth_path, exist_ok=True)
        save_settings(self.args)

    def save_mesh(self, name='fused_final.ply'):
        target_path = os.path.join(self.args.fused_mesh_path, name)

        save_mesh(
            vertices=self.vertices,
            faces=self.faces,
            colors=self.colors,
            target_path=target_path
        )

        return target_path

    def load_mesh(self, rgb_path):
        vertices, faces, rgb = load_mesh(rgb_path)
        self.vertices = vertices.to(self.vertices)
        self.colors = rgb.to(self.colors)
        self.faces = faces.to(self.faces)

    def predict_depth(self, inpainted_image_pils, fix_input_depth=True):
        # use the IronDepth method to predict depth: https://github.com/baegwangbin/IronDepth

        predicted_depths = []
        predicted_norms = []
            
        for inpainted_image_pil in inpainted_image_pils:
            if self.args.gaussian_blur_before_predict_depth:
                image_pil = Image.fromarray(cv2.GaussianBlur(cv2.bilateralFilter(np.array(inpainted_image_pil), 5, 140, 140), (5,5), 0))
            else:
                image_pil = inpainted_image_pil

            predicted_depth, predicted_norm = predict_iron_depth(
                image=image_pil,
                K=self.K,
                device=self.args.device,
                model=self.iron_depth_model,
                n_net=self.iron_depth_n_net,
                input_depth=self.rendered_depth,
                input_mask=self.inpaint_mask,
                fix_input_depth=fix_input_depth
            )

            predicted_depths.append(predicted_depth)
            predicted_norms.append(predicted_norm)

        return predicted_depths, predicted_norms

    def depth_alignment(self, predicted_depths, fuse=True):

        aligned_depths = []
        
        for predicted_depth in predicted_depths:
            aligned_depth = depth_alignment.scale_shift_linear(
                rendered_depth=self.rendered_depth,
                predicted_depth=predicted_depth,
                mask=~self.inpaint_mask,
                fuse=fuse)
            aligned_depths.append(aligned_depth)

        return aligned_depths
    
    def active_picking(self, inpainted_image_pils, predicted_depths, predicted_norms, aligned_depths):

        if self.args.pick_by_variance:
            idx = 0
            min = 0
            for i, aligned_depth in enumerate(aligned_depths):
                depth_norm = aligned_depth[self.inpaint_mask].norm()
                if depth_norm > min:
                    idx = i
                    min = depth_norm
        else:
            idx = 0
            max = 0
            for i, aligned_depth in enumerate(aligned_depths):
                depth_mean = aligned_depth[self.inpaint_mask].mean()
                if depth_mean > max:
                    idx = i
                    max = depth_mean

        return inpainted_image_pils[idx], predicted_depths[idx], predicted_norms[idx], aligned_depths[idx]

    def add_vertices_and_faces(self, inpainted_image, predicted_depth, use_opencv_camera_with_intrinsic=None):
        if self.inpaint_mask.sum() == 0:
            # when no pixels were masked out, we do not need to add anything, so skip this call
            return

        vertices, faces, colors = features_to_world_space_mesh(
            colors=inpainted_image,
            depth=predicted_depth,
            fov_in_degrees=self.args.fov,
            world_to_cam=self.world_to_cam,
            mask=self.inpaint_mask,
            edge_threshold=self.args.edge_threshold,
            surface_normal_threshold=self.args.surface_normal_threshold,
            pix_to_face=self.pix_to_face,
            faces=self.faces,
            vertices=self.vertices,
            use_opencv_camera_with_intrinsic=use_opencv_camera_with_intrinsic
        )

        faces += self.vertices.shape[1]  # add face offset

        self.vertices = torch.cat([self.vertices, vertices], dim=1)
        self.colors = torch.cat([self.colors, colors], dim=1)
        self.faces = torch.cat([self.faces, faces], dim=1)

    def remove_masked_out_faces(self, remove_faces_in_front=False):
        # if remove_faces_in_front is False, only remove the both sides of faces in the range of the threshold.
        # if remove_faces_in_front is True, also remove all the faces in front of the new pixels.

        if self.pix_to_face is None:
            return

        # get faces to remove: those faces that project into the inpaint_mask
        faces_to_remove = self.pix_to_face[:, self.inpaint_mask, :]

        # only remove faces whose depth is close to actual depth
        if self.args.remove_faces_depth_threshold > 0:
            depth = self.rendered_depth[self.inpaint_mask]
            depth = depth[None, ..., None]
            depth = depth.repeat(faces_to_remove.shape[0], 1, faces_to_remove.shape[-1])
            zbuf = self.z_buf[:, self.inpaint_mask, :]
            if not remove_faces_in_front:
                mask_zbuf = (zbuf - depth).abs() < self.args.remove_faces_depth_threshold
            else:
                mask_zbuf = (zbuf < depth) | ((zbuf - depth).abs() < self.args.remove_faces_depth_threshold)
            faces_to_remove = faces_to_remove[mask_zbuf]

        faces_to_remove = torch.unique(faces_to_remove.flatten())
        faces_to_remove = faces_to_remove[faces_to_remove > -1].long()

        # select the faces that were hit in the mask
        # this does not catch all faces because some faces that project into the mask are not visible from current viewpoint (e.g. behind another face)
        # this _should not_ catch those faces though - they might not be wanted to be removed.
        keep_faces_mask = torch.ones_like(self.faces[0], dtype=torch.bool)
        keep_faces_mask[faces_to_remove] = False

        # remove the faces
        self.faces = self.faces[:, keep_faces_mask]

        # remove left-over too long faces
        self.apply_edge_threshold_filter()

        # set to None since pix_to_face has now changed
        # this is actually desired behavior: we do not fuse together new faces with current mesh, because it is too difficult anyways
        self.pix_to_face = None

    def set_simple_trajectory_poses(self, poses):
        self.simple_trajectory_poses = poses
        self.args.n_images = len(poses)
        

    def project(self, use_opencv_camera_with_intrinsic=None, force_cull_backfaces=False):
        # project mesh into pose and render (rgb, depth, mask)
        rendered_image_tensor, self.rendered_depth, self.inpaint_mask, self.pix_to_face, self.z_buf, self.backface_mask = render_mesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_features=self.colors,
            H=self.H,
            W=self.W,
            fov_in_degrees=self.args.fov,
            RT=self.world_to_cam,
            blur_radius=self.args.blur_radius,
            faces_per_pixel=self.args.faces_per_pixel, 
            use_opencv_camera_with_intrinsic=use_opencv_camera_with_intrinsic,
            cull_backfaces=self.args.mask_backward_facing_surface or force_cull_backfaces
            
        )

        # mask rendered_image_tensor
        rendered_image_tensor = rendered_image_tensor * ~self.inpaint_mask

        # stable diffusion models want the mask and image as PIL images
        rendered_image_pil = Image.fromarray((rendered_image_tensor.permute(1, 2, 0).detach().cpu().numpy()[..., :3] * 255).astype(np.uint8))
        inpaint_mask_pil = Image.fromarray(self.inpaint_mask.detach().cpu().squeeze().float().numpy() * 255).convert("RGB")

        return rendered_image_tensor, rendered_image_pil, inpaint_mask_pil

    def inpaint(self, rendered_image_pil, inpaint_mask_pil, num_sampling):
        m = np.asarray(inpaint_mask_pil)[..., 0].astype(np.uint8)

        # inpaint with classical method to fill small gaps
        rendered_image_numpy = np.asarray(rendered_image_pil)
        rendered_image_pil = Image.fromarray(cv2.inpaint(rendered_image_numpy, m, 3, cv2.INPAINT_TELEA))

        # remove small seams from mask
        m2 = m
        if self.args.erode_iters > 0:
            m2 = cv2.erode(m, (3, 3), iterations=self.args.erode_iters)
        if self.args.dilate_iters > 0:
            for _ in range(self.args.dilate_iters):
                m2 = cv2.GaussianBlur(m2, (7, 7), 0)
        m2[m2!=0] = -1 # set maximum if not zero

        # convert back to pil & save updated mask
        inpaint_mask_pil = Image.fromarray(m2).convert("RGB")
        self.eroded_dilated_inpaint_mask = torch.from_numpy(m2).to(self.inpaint_mask)

        # update inpaint mask to contain all updates
        if self.args.update_mask_after_improvement:
            self.inpaint_mask = self.inpaint_mask + self.eroded_dilated_inpaint_mask

        if self.args.blur_before_inpainting:
            rendered_image_pil = Image.fromarray(cv2.bilateralFilter(cv2.GaussianBlur(np.array(rendered_image_pil), (5,5), 0), 7, 75, 75))

        # inpaint large missing areas with stable-diffusion model
        inpainted_image_pils = []
        for _ in range(num_sampling):
            inpainted_image_pil = self.diffusion_pipe.inpaint_perspective_image(
                prompt=self.args.prompt,
                negative_prompt=self.args.negative_prompt,
                num_images_per_prompt=1,
                image=rendered_image_pil,
                mask_image=inpaint_mask_pil,
                guidance_scale=self.args.guidance_scale,
                num_inference_steps=self.args.num_inference_steps
            ).images[0]
            inpainted_image_pils.append(inpainted_image_pil)

        return inpainted_image_pils, inpaint_mask_pil
    

    def apply_depth_smoothing(self, image, mask):
        # mainly gapping the sharp edges between predicted depth and gt_depth

        def sobel(x):
            flipped_sobel_x = torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=torch.float32).to(x.device)
            flipped_sobel_x = torch.stack([flipped_sobel_x, flipped_sobel_x.t()]).unsqueeze(1)

            x_pad = torch.nn.functional.pad(x.float()[None, None, ...], (1, 1, 1, 1), mode="replicate")

            x = torch.nn.functional.conv2d(
                x_pad,
                flipped_sobel_x,
                padding="valid"
            )
            dx, dy = x.unbind(dim=-3)
            return torch.sqrt(dx**2 + dy**2).squeeze() > 0

        edges = sobel(mask).float().cpu().numpy()
        edges = torch.from_numpy(cv2.GaussianBlur(edges, (5, 5), 0)).to(mask)

        img_numpy = image.float().cpu().numpy()
        blur_bilateral = cv2.bilateralFilter(img_numpy, 5, 140, 140)
        blur_gaussian = cv2.GaussianBlur(blur_bilateral, (5, 5), 0)
        blur_gaussian = torch.from_numpy(blur_gaussian).to(image)

        image_smooth = torch.where(edges, blur_gaussian, image)
        return image_smooth
    
    def apply_depth_edge_masking(self, predicted_depth, inpaint_mask):
        # mask out the edges in the predict depth map, 
        # because the predicted depth values around the edge is not sharp enough (which cause weird mesh edges).

        # filter points that don't have depth values. (depth value will be 0 if not available)
        inpaint_mask = inpaint_mask & (predicted_depth>0)

        # filter depth edges (most ambiguous depth values around edges cause weird mesh edges)
        depth_gray = (predicted_depth/predicted_depth.max()*255).cpu().numpy().astype('uint8')

        depth_gray_mean = depth_gray[depth_gray!=0].mean()
        low_threshold = 0.15 * depth_gray_mean
        high_threshold = 0.45 * depth_gray_mean
        edges = cv2.Canny(depth_gray, low_threshold, high_threshold)
        edges = cv2.GaussianBlur(edges, (7, 7), 0)

        edge_mask = torch.tensor(edges!=0).to(inpaint_mask)
        inpaint_mask = inpaint_mask & ~edge_mask

        predicted_depth_edge_masked = predicted_depth.clone()
        predicted_depth_edge_masked[edge_mask] = 0

        return predicted_depth, predicted_depth_edge_masked, inpaint_mask

    def add_next_image(self, inpainted_image_pils, pos, offset, save_files=True, file_suffix="", gt_depth=None, use_opencv_camera_with_intrinsic=None):        
        if gt_depth is not None:
            # backup the origin mask where we want to inpaint
            inpaint_mask_origin = self.inpaint_mask

            # replace rendered_depth as gt_depth if availale
            self.rendered_depth = gt_depth

            # inpaint lost values in gt_depth 
            self.inpaint_mask = (gt_depth==0)

        # predict & align depth of current image
        # refine all the depth values based on gt_depth if having gt_depth (we found out that it cause the smoother depth)
        if gt_depth is not None and self.args.use_only_gt_depth:
            predicted_depths = [gt_depth] * len(inpainted_image_pils)
            aligned_depths = [gt_depth] * len(inpainted_image_pils)
            predicted_norms = [None] * len(inpainted_image_pils)
        else:
            predicted_depths, predicted_norms = self.predict_depth(inpainted_image_pils, fix_input_depth=True)
            aligned_depths = predicted_depths# = self.depth_alignment(predicted_depths, gt_depth is None)
        
        # pick a best inpainting result
        inpainted_image_pil, predicted_depth, predicted_norm, aligned_depth = self.active_picking(inpainted_image_pils, predicted_depths, predicted_norms, aligned_depths)
        inpainted_image = pil_to_torch(inpainted_image_pil, self.args.device)
        
        if gt_depth is not None:
            if self.args.use_only_gt_depth:
                # if not inpaint init depth, use this instead
                self.inpaint_mask = (gt_depth!=0)
                aligned_depth = gt_depth
            else:
                # return back to the origin mask where we want to inpaint
                self.inpaint_mask = inpaint_mask_origin

                if self.args.mask_inpainted_depth_edges:
                    aligned_depth, aligned_depth_edge_masked, self.inpaint_mask = self.apply_depth_edge_masking(aligned_depth, self.inpaint_mask)

            # remove overlapping pixels of the given images.
            if self.args.replace_overlapping and self.vertices.shape[1]!=0:
                _, _, overlapping_mask, self.pix_to_face, self.z_buf, _ = render_mesh(
                    vertices=self.vertices,
                    faces=self.faces,
                    vertex_features=self.colors,
                    H=self.H,
                    W=self.W,
                    fov_in_degrees=self.args.fov,
                    RT=self.world_to_cam,
                    blur_radius=self.args.blur_radius,
                    faces_per_pixel=self.args.faces_per_pixel, 
                    use_opencv_camera_with_intrinsic=use_opencv_camera_with_intrinsic,
                    cull_backfaces=self.args.mask_backward_facing_surface
                )
                self.remove_masked_out_faces()#remove_faces_in_front=True)
        else:
            if self.args.mask_inpainted_depth_edges:
                aligned_depth, aligned_depth_edge_masked, self.inpaint_mask = self.apply_depth_edge_masking(aligned_depth, self.inpaint_mask)
            else:
                aligned_depth = self.apply_depth_smoothing(aligned_depth, self.inpaint_mask)

            # remove masked-out faces. If we use erosion in the mask it means those points will be removed.
            # it may cause holes if the scene is complex
            if self.args.replace_over_inpainted:
                # only now update mask: predicted depth will still take old positions as anchors (they are still somewhat correct)
                # otherwise if we erode/dilate too much we could get depth estimates that are way off
                if not self.args.update_mask_after_improvement:
                    self.inpaint_mask = self.inpaint_mask + self.eroded_dilated_inpaint_mask

                self.remove_masked_out_faces()

        # add new points (novel content)
        self.add_vertices_and_faces(inpainted_image, aligned_depth, use_opencv_camera_with_intrinsic)

        if save_files and self.args.save_files:
            depth_pil = Image.fromarray(visualize_depth_numpy(aligned_depth.cpu().numpy())[0].astype(np.uint8))
            normal_pil = Image.fromarray(normal_to_rgb(predicted_norm.cpu().numpy()))
            depth_pil_not_aligned = Image.fromarray(visualize_depth_numpy(predicted_depth.cpu().numpy())[0].astype(np.uint8))
            save_image(inpainted_image_pil, f"rgb{file_suffix}", offset + pos, self.args.rgb_path)
            save_image(depth_pil, f"depth{file_suffix}", offset + pos, self.args.depth_path)
            save_image(normal_pil, f"normal{file_suffix}", offset + pos, self.args.depth_path)
            #save_image(depth_pil_not_aligned, f"depth_not_aligned{file_suffix}", offset + pos, self.args.depth_path)
            #save_rgbd(inpainted_image_pil, depth_pil, f'rgbd{file_suffix}', offset + pos, self.args.rgbd_path)
            if gt_depth is not None:
                # depth_edge_masked_pil = Image.fromarray(visualize_depth_numpy(aligned_depth_edge_masked.cpu().numpy())[0].astype(np.uint8))
                # save_image(depth_edge_masked_pil, f"depth_edge_masked{file_suffix}", offset + pos, self.args.depth_path)
                gt_depth_pil = Image.fromarray(visualize_depth_numpy(gt_depth.cpu().numpy())[0].astype(np.uint8))
                save_image(gt_depth_pil, f"gt_depth{file_suffix}", offset + pos, self.args.depth_path)

        # save current meshes
        if save_files and self.args.save_files and self.args.save_scene_every_nth > 0 and (offset + pos) % self.args.save_scene_every_nth == 0:
            self.save_mesh(f"fused_until_frame{file_suffix}_{offset + pos:04}.ply")

        return inpainted_image_pil, inpainted_image, predicted_depth, predicted_norm, aligned_depth

    def project_and_inpaint_with_poses(self, pos=0, offset=0, intrinsic=None, save_files=True, file_suffix=""):
        # project to next pose
        _, rendered_image_pil, inpaint_mask_pil  = self.project(use_opencv_camera_with_intrinsic=intrinsic)

        # inpaint projection result
        inpainted_image_pils, eroded_dilated_inpaint_mask_pil = self.inpaint(rendered_image_pil, inpaint_mask_pil, self.args.num_inpainting_sampling)
        if save_files and self.args.save_files:
            save_image(eroded_dilated_inpaint_mask_pil, f"mask_eroded_dilated{file_suffix}", offset + pos, self.args.mask_path)

        # predict depth, add to 3D structure
        inpainted_image_pil, _, _, _, predicted_depth = self.add_next_image(inpainted_image_pils=inpainted_image_pils, pos=pos, offset=offset, \
                                                                            save_files=save_files, file_suffix=file_suffix, use_opencv_camera_with_intrinsic=self.K)

        # update images
        self.predicted_depth = predicted_depth

        if save_files and self.args.save_files:
            rendered_depth_pil = Image.fromarray(visualize_depth_numpy(self.rendered_depth.cpu().numpy())[0].astype(np.uint8))
            save_image(rendered_image_pil, f"rendered_rgb{file_suffix}", offset + pos, self.args.rendered_rgb_path)
            save_image(rendered_depth_pil, f"rendered_depth{file_suffix}", offset + pos, self.args.rendered_depth_path)
            save_image(inpaint_mask_pil, f"mask{file_suffix}", offset + pos, self.args.mask_path)
            #save_rgbd(rendered_image_pil, rendered_depth_pil, f'rendered_rgbd{file_suffix}', offset + pos, self.args.rendered_rgbd_path)

        # update bounding box
        self.calc_bounding_box()
    
    def project_and_inpaint(self, pos=0, offset=0, intrinsic=None, save_files=True, file_suffix=""):
        # project to next pose
        _, rendered_image_pil, inpaint_mask_pil  = self.project(use_opencv_camera_with_intrinsic=intrinsic)

        # inpaint projection result
        inpainted_image_pils, eroded_dilated_inpaint_mask_pil = self.inpaint(rendered_image_pil, inpaint_mask_pil, self.args.num_inpainting_sampling)
        if save_files and self.args.save_files:
            save_image(eroded_dilated_inpaint_mask_pil, f"mask_eroded_dilated{file_suffix}", offset + pos, self.args.mask_path)

        # predict depth, add to 3D structure
        inpainted_image_pil, _, _, _, predicted_depth = self.add_next_image(inpainted_image_pils=inpainted_image_pils, pos=pos, offset=offset, \
                                                                            save_files=save_files, file_suffix=file_suffix)

        # update images
        self.predicted_depth = predicted_depth

        if save_files and self.args.save_files:
            rendered_depth_pil = Image.fromarray(visualize_depth_numpy(self.rendered_depth.cpu().numpy())[0].astype(np.uint8))
            save_image(rendered_image_pil, f"rendered_rgb{file_suffix}", offset + pos, self.args.rendered_rgb_path)
            save_image(rendered_depth_pil, f"rendered_depth{file_suffix}", offset + pos, self.args.rendered_depth_path)
            save_image(inpaint_mask_pil, f"mask{file_suffix}", offset + pos, self.args.mask_path)
            #save_rgbd(rendered_image_pil, rendered_depth_pil, f'rendered_rgbd{file_suffix}', offset + pos, self.args.rendered_rgbd_path)

        # update bounding box
        self.calc_bounding_box()

    def clean_mesh(self):
        self.vertices, self.faces, self.colors = clean_mesh(
            vertices=self.vertices,
            faces=self.faces,
            colors=self.colors,
            edge_threshold=self.args.edge_threshold,
            min_triangles_connected=self.args.min_triangles_connected,
            outlier_point_radius=self.args.outlier_point_radius,
            fill_holes=True
        )

        self.vertices, self.faces, self.colors = simplify_mesh(self.vertices, self.faces, self.colors, self.args.simplify_mesh_voxel_size)

    def apply_edge_threshold_filter(self):
        self.faces = edge_threshold_filter(
            vertices=self.vertices,
            faces=self.faces,
            edge_threshold=self.args.edge_threshold
        )

    def render_with_pose(self, intrinsic, pose):

        self.world_to_cam = pose

        _, rendered_image_pil, inpaint_mask_pil = self.project(use_opencv_camera_with_intrinsic=intrinsic)

        rendered_image = np.array(rendered_image_pil)

        rendered_depth = self.rendered_depth.cpu().numpy()

        mask = (np.array(inpaint_mask_pil)==255)[...,0]
        
        # reset gpu memory
        torch.cuda.empty_cache()

        return rendered_image, rendered_depth, mask
    
    def render_with_pose_pil(self, intrinsic, pose, rotate=False, depth_minmax=None):

        self.world_to_cam = pose
        # project to next pose
        _, rendered_image_pil, inpaint_mask_pil  = self.project(use_opencv_camera_with_intrinsic=intrinsic)
        rendered_depth_pil = Image.fromarray(visualize_depth_numpy(self.rendered_depth.cpu().numpy(), minmax=depth_minmax)[0].astype(np.uint8))
        
        if rotate:
            rendered_image_pil = Image.fromarray(np.array(rendered_image_pil)[::-1,::-1,:])
            rendered_depth_pil = Image.fromarray(np.array(rendered_depth_pil)[::-1,::-1,:])
            inpaint_mask_pil = Image.fromarray(np.array(inpaint_mask_pil)[::-1,::-1,:])

        rendered_rgbd_pil = get_rgbd_image(rendered_image_pil, rendered_depth_pil)

        return rendered_image_pil, rendered_depth_pil, inpaint_mask_pil, rendered_rgbd_pil

    def forward_with_poses(self, pos=0, offset=0, save_files=True):
        # get next pose
        self.world_to_cam = torch.tensor(self.simple_trajectory_poses[pos], device=self.args.device)

        # render --> inpaint --> add to 3D structure
        self.project_and_inpaint_with_poses(pos, offset, self.K, save_files)

        if self.args.clean_mesh_every_nth > 0 and (pos + offset) % self.args.clean_mesh_every_nth == 0:
            self.clean_mesh()

    def refine(self, pos=0, offset=0, repeat_iters=1):
        # save old values
        #old_min_triangles_connected = self.args.min_triangles_connected
        #old_surface_normal_threshold = self.args.surface_normal_threshold

        #self.args.min_triangles_connected = -1
        #self.args.surface_normal_threshold = -1
        

        self.project_and_inpaint(pos, offset, file_suffix=f"_refine")


        # repeat to fill in remaining holes
        if self.args.clean_mesh_every_nth > 0 and (pos + offset) % self.args.clean_mesh_every_nth == 0:
            self.clean_mesh()

        # reset to old values
        #self.args.min_triangles_connected = old_min_triangles_connected
        #self.args.surface_normal_threshold = old_surface_normal_threshold

    def generate_images_with_poses(self, offset=0):
        # generate images with forward-warping
        pbar = tqdm(range(self.args.n_images))
        for pos in pbar:
            pbar.set_description(f"Image [{pos}/{self.args.n_images - 1}]")
            self.forward_with_poses(pos, offset)

        # reset gpu memory
        torch.cuda.empty_cache()

        return offset + self.args.n_images

    def calc_bounding_box(self):
        """
        Calculate the bounding box of existing meshes. 
        We use the most simply version to calculate: [x_min, y_min, z_min] -> [x_max, y_max, z_max]
        """
        min_bound = torch.amin(self.vertices, dim=-1)
        max_bound = torch.amax(self.vertices, dim=-1)
        self.bbox = [min_bound, max_bound]

    def sample_random_point_for_completion(self):
        '''
        return a random point in the scene bounding box scaled by self.args.core_ratios
        '''
        self.calc_bounding_box()
        min_bound, max_bound = self.bbox
        core_ratio = torch.tensor([self.args.core_ratio_x, self.args.core_ratio_y, self.args.core_ratio_z]).to(min_bound.device)
        scale = (max_bound - min_bound) * core_ratio
        shift = min_bound + (max_bound - min_bound) * (torch.ones_like(core_ratio) - core_ratio) / 2
        pos = torch.rand(3).to(min_bound.device) * scale + shift

        return pos

    
    def calculate_completion_criterion(self):
        '''
        need to first setup self.world_to_cam
        '''
        _, rendered_image_pil, inpaint_mask_pil = self.project(force_cull_backfaces=True)
        inpaint_mask = np.array(inpaint_mask_pil)[..., 0] / 255.0
        inpaint_ratio = np.sum(inpaint_mask) / self.H / self.W
        backface_ratio = torch.sum(self.backface_mask) / self.H / self.W
        if torch.sum(self.rendered_depth) != 0:
            min_depth = torch.min(self.rendered_depth[self.rendered_depth!=0]).cpu().numpy()
        else:
            min_depth = 0
        
        return inpaint_ratio, backface_ratio, min_depth


    def complete_mesh(self, offset=0):
        '''
        stategy:
        during each iteration, try to find a best view from 400 randomly sampled views:
        1) camera positon must be at least 1m away from meshes
        2) inpaint ratio must be less than 50% (inpaint ratio = inpaint area / image area)
        3) backface ratio must be less than 1%
        4) minimum depth is greater than 1m
        5) pick the one with greatest criterion (inpaint ratio * minimum depth). (want to inpint big holes first)
        6) after a view is picked, move the camera backward as long as criteria 2~4 are satisfied
        '''
        pos = 0
        
        original_num_inpainting_sampling = self.args.num_inpainting_sampling
        self.args.num_inpainting_sampling = self.args.num_inpainting_sampling_completion

        for i in tqdm(range(self.args.complete_mesh_iter)):
            # switch to lower rendering resolution to speed up sampling
            original_H = self.H
            original_W = self.W
            original_fov = self.args.fov
            original_K = self.K
            self.H = self.args.completion_sample_resolution
            self.W = self.args.completion_sample_resolution
            self.args.fov = original_fov
            self.K = get_pinhole_intrinsics_from_fov(self.H, self.W, self.args.fov).to(self.K)

            inpaint_ratios = []
            RT = []
            picking_criterion = []
            
            for j in range(self.args.complete_mesh_num_sample_per_iter):
                # sample from random pos in grid_xyz and random camera orientation
                camera_pos = self.sample_random_point_for_completion()
                theta = torch.rand(1) * 360
                phi = torch.rand(1) * self.args.completion_camera_elevation_angle_limit * 2 - self.args.completion_camera_elevation_angle_limit #(30 if i < self.args.complete_mesh_iter / 2 else -30)
                c2w = trajectory_util.pose_spherical(theta, phi, 1).to(self.args.device)
                c2w[0:3, 3] = camera_pos
                self.world_to_cam = torch.inverse(c2w)

                inpaint_ratio, backface_ratio, min_depth = self.calculate_completion_criterion()
                
                if inpaint_ratio <= self.args.max_inpaint_ratio and inpaint_ratio >= self.args.min_inpaint_ratio and backface_ratio < self.args.max_backface_ratio and min_depth > self.args.min_depth_quantil_to_mesh:
                    RT.append(self.world_to_cam)
                    inpaint_ratios.append(inpaint_ratio)
                    picking_criterion.append(inpaint_ratio * min_depth)

                    
            # switch back to normal resolution
            self.H = original_H
            self.W = original_W
            self.args.fov = original_fov
            self.K = original_K
                
            if len(inpaint_ratios) > 0:
                # select the optimal view by criterion
                decision = np.argmax(picking_criterion)

                # for the selected sample view, step back
                self.world_to_cam = RT[decision].to(self.args.device)
                c2w = torch.inverse(self.world_to_cam)
                step_back_vector = c2w[0:3, 2].clone()
                camera_pos = c2w[0:3, 3].clone()
                num_step = 0
                
                inpaint_ratio, backface_ratio, min_depth = self.calculate_completion_criterion()
                while inpaint_ratio <= self.args.max_inpaint_ratio and inpaint_ratio >= self.args.min_inpaint_ratio and backface_ratio < self.args.max_backface_ratio and min_depth > self.args.min_depth_quantil_to_mesh:
                    num_step += 1
                    c2w[0:3, 3] = camera_pos - step_back_vector * num_step * self.args.complete_mesh_step_back_length
                    self.world_to_cam = torch.inverse(c2w)
                    inpaint_ratio, backface_ratio, min_depth = self.calculate_completion_criterion()
                
                # finally starts inpainting
                c2w[0:3, 3] = camera_pos - step_back_vector * (num_step - 1) * self.args.complete_mesh_step_back_length
                self.world_to_cam = torch.inverse(c2w)
                self.refine(pos, offset, repeat_iters=0)
                pos += 1
                torch.cuda.empty_cache()
        
        self.args.num_inpainting_sampling = original_num_inpainting_sampling
        
        return offset + pos
        


    def project_cubemap(self, center_pos=None, pos=0, offset=0, save_files=True, file_suffix="", inpainted_images=None, predicted_depths=None, inpaint_masks_input=None, inpainted_images_gt=None, predicted_depths_gt=None, inpaint_masks_input_gt=None):
        original_K = self.K
        original_fov = self.args.fov
        self.K = get_pinhole_intrinsics_from_fov(self.H, self.W, 90).to(self.world_to_cam)
        self.args.fov = 90

        H_pano = 1024
        W_pano = 2048

        self.cubemap_world_to_cams = get_cubemap_w2c(center_pos)
        
        rendered_images = []
        inpaint_masks_cubes = []
        rendered_depths = []
        depth_max = 0

        for i, world_to_cam in enumerate(self.cubemap_world_to_cams):
            self.world_to_cam = world_to_cam
            _, rendered_image_pil, inpaint_mask_pil = self.project()
            
            rendered_images.append(pil_to_torch(rendered_image_pil, self.args.device))
            inpaint_masks_cubes.append(pil_to_torch(inpaint_mask_pil, self.args.device))
            rendered_depths.append((self.rendered_depth * depth_to_distance_ratio(self.H, self.W, self.K)).unsqueeze(0).expand(3, -1, -1))
            if depth_max < (self.rendered_depth * depth_to_distance_ratio(self.H, self.W, self.K)).max():
                depth_max = (self.rendered_depth * depth_to_distance_ratio(self.H, self.W, self.K)).max()

            save_image(rendered_image_pil, f"rendered_rgb_cubemap{file_suffix}", offset + i, self.args.rendered_rgb_path)
        
        
        rendered_images_equi = cube2equi(cubemap=rendered_images, cube_format="list", height=H_pano, width=W_pano)
        rendered_images = rendered_images_equi[:, 256:768, :]    # 90: 256:768 60: 512:1024
        inpaint_masks_equi = cube2equi(cubemap=inpaint_masks_cubes, cube_format="list", height=H_pano, width=W_pano)
        inpaint_masks = inpaint_masks_equi[:, 256:768, :]
        rendered_depths = cube2equi(cubemap=[i/depth_max for i in rendered_depths], cube_format="list", height=H_pano, width=W_pano)[:, 256:768, :]
        #inpaint_masks[inpaint_masks != 0] = 1
        #inpaint_masks.to(torch.bool)
        rendered_depths[inpaint_masks != 0] = 0
        rendered_depths = rendered_depths[0, ...] * depth_max
        

        to_pil = transforms.ToPILImage()
        save_image(to_pil(rendered_images), f"rendered_rgb_pano{file_suffix}", offset, self.args.rendered_rgb_path)
        save_image(to_pil(inpaint_masks), f"mask_pano{file_suffix}", offset, self.args.mask_path)
        rendered_depth_pil = Image.fromarray(visualize_depth_numpy(rendered_depths.cpu().numpy())[0].astype(np.uint8))
        save_image(rendered_depth_pil, f"rendered_depth_pano{file_suffix}", offset + pos, self.args.rendered_depth_path)

        self.K = original_K
        self.args.fov = original_fov
    
        return offset + 1

    
    def project_and_inpaint_panorama(self, center_pos=None, pos=0, offset=0, save_files=True, file_suffix=""):
        original_fov = self.args.fov
        original_K = self.K
        self.args.fov = 90
        self.K = get_pinhole_intrinsics_from_fov(self.H, self.W, 90).to(self.world_to_cam)

        H_pano = 1024
        W_pano = 2048
        
        rendered_image_cubemap = []
        inpaint_mask_cubemap = []
        rendered_depths = []
        depth_max = 0

        self.cubemap_world_to_cams = get_cubemap_w2c(center_pos)

        for i, world_to_cam in enumerate(self.cubemap_world_to_cams):
            self.world_to_cam = world_to_cam
            _, rendered_image_pil, inpaint_mask_pil = self.project()
            
            rendered_image_cubemap.append(pil_to_torch(rendered_image_pil, self.args.device))
            inpaint_mask_cubemap.append(pil_to_torch(inpaint_mask_pil, self.args.device))
            rendered_depths.append((self.rendered_depth * depth_to_distance_ratio(self.H, self.W, self.K)).unsqueeze(0).expand(3, -1, -1))
            if depth_max < (self.rendered_depth * depth_to_distance_ratio(self.H, self.W, self.K)).max():
                depth_max = (self.rendered_depth * depth_to_distance_ratio(self.H, self.W, self.K)).max()

            #save_image(rendered_image_pil, f"rendered_rgb_cubemap{file_suffix}", offset + i, self.args.rendered_rgb_path)
        
        
        rendered_images_equi = cube2equi(cubemap=rendered_image_cubemap, cube_format="list", height=H_pano, width=W_pano)
        rendered_images = rendered_images_equi[:, 256:768, :]    # 90: 256:768 60: 512:1024
        inpaint_masks_equi = cube2equi(cubemap=inpaint_mask_cubemap, cube_format="list", height=H_pano, width=W_pano)
        inpaint_masks = inpaint_masks_equi[:, 256:768, :]
        rendered_depths = cube2equi(cubemap=[i/depth_max for i in rendered_depths], cube_format="list", height=H_pano, width=W_pano)[:, 256:768, :]
        #inpaint_masks[inpaint_masks != 0] = 1
        #inpaint_masks.to(torch.bool)
        rendered_depths[inpaint_masks != 0] = 0
        rendered_depths = rendered_depths[0, ...] * depth_max

        

        to_pil = transforms.ToPILImage()
        save_image(to_pil(rendered_images), f"rendered_rgb_pano{file_suffix}", offset, self.args.rendered_rgb_path)
        save_image(to_pil(inpaint_masks), f"mask_pano{file_suffix}", offset, self.args.mask_path)
        rendered_depth_pil = Image.fromarray(visualize_depth_numpy(rendered_depths.cpu().numpy())[0].astype(np.uint8))
        save_image(rendered_depth_pil, f"rendered_depth_pano{file_suffix}", offset + pos, self.args.rendered_depth_path)


        
        inpainted_image = self.diffusion_pipe.inpaint_panorama(image_cubemap=rendered_image_cubemap, inpaint_mask_cubemap=inpaint_mask_cubemap, prompt=self.args.prompt, negative_prompt=self.args.negative_prompt)
        save_image(to_pil(inpainted_image), f"rgb_pano_sample{file_suffix}", offset, self.args.rgb_path)
        

        # predict depth, add to 3D structure
        self.rendered_depth = rendered_depths
        self.inpaint_mask = inpaint_masks[0, ...].to(torch.bool)
        
        inpainted_image_pil, _, _, _, predicted_depth = self.add_next_panorama(inpainted_image_pil=to_pil(inpainted_image), inpaint_masks_cubes=inpaint_mask_cubemap, H_pano=H_pano, W_pano=W_pano, pos=pos, offset=offset, save_files=save_files, file_suffix=file_suffix)
        
        # update bounding box
        self.calc_bounding_box()

        self.args.fov = original_fov
        self.K = original_K
    
        return offset + 1

    def add_next_panorama(self, inpainted_image_pil, inpaint_masks_cubes, H_pano, W_pano, pos, offset, aligned_depth=None, save_files=True, file_suffix="", gt_depth=None, use_opencv_camera_with_intrinsic=None):        
        inpainted_image_equi = torch.zeros((3, H_pano, W_pano), device=self.args.device,dtype=torch.float32)
        inpainted_image_equi[:, 256:768, :] = pil_to_torch(inpainted_image_pil, device = self.args.device)

        inpaint_mask_pano = self.inpaint_mask.to(torch.bool)
        inpaint_mask_equi = torch.ones((H_pano, W_pano), device=self.args.device, dtype=self.inpaint_mask.dtype)
        inpaint_mask_equi[256:768, :] = self.inpaint_mask
        self.inpaint_mask = inpaint_mask_equi.to(torch.bool)
        

        rendered_depth_pano = self.rendered_depth
        rendered_depth_equi = torch.zeros((H_pano, W_pano), device=self.args.device, dtype=self.rendered_depth.dtype)
        rendered_depth_equi[256:768, :] = self.rendered_depth
        self.rendered_depth = rendered_depth_equi

        if gt_depth is not None:
            # backup the origin mask where we want to inpaint
            inpaint_mask_origin = self.inpaint_mask

            # replace rendered_depth as gt_depth if availale
            self.rendered_depth = gt_depth

            # inpaint lost values in gt_depth 
            self.inpaint_mask = (gt_depth==0)

        # predict & align depth of current image
        # refine all the depth values based on gt_depth if having gt_depth (we found out that it cause the smoother depth)
        predicted_depth, predicted_norm = predict_iron_depth_panorama(
            image=inpainted_image_equi,
            K=get_pinhole_intrinsics_from_fov(512, 512, 90).to(self.world_to_cam),
            device=self.args.device,
            model=self.iron_depth_model,
            n_net=self.iron_depth_n_net,
            input_depth=self.rendered_depth,
            input_mask=self.inpaint_mask,
            fix_input_depth=True,
            num_views=32
        )

        predicted_depth = predicted_depth[256:768, :]
        self.rendered_depth = rendered_depth_pano
        self.inpaint_mask = inpaint_mask_pano
        
        # pick a best inpainting result
        inpainted_image_pil = inpainted_image_pil
        aligned_depth = predicted_depth
        inpainted_image = pil_to_torch(inpainted_image_pil, self.args.device)

        if gt_depth is None:
            aligned_depth = self.apply_depth_smoothing(aligned_depth, self.inpaint_mask)
        
        if gt_depth is not None:
            # return back to the origin mask where we want to inpaint
            self.inpaint_mask = inpaint_mask_origin

            # if not inpaint init depth, use this instead
            # self.inpaint_mask = (gt_depth!=0)

        if save_files and self.args.save_files:
            depth_pil = Image.fromarray(visualize_depth_numpy(aligned_depth.cpu().numpy())[0].astype(np.uint8))
            normal_pil = Image.fromarray(normal_to_rgb(predicted_norm.cpu().numpy()))
            depth_pil_not_aligned = Image.fromarray(visualize_depth_numpy(predicted_depth.cpu().numpy())[0].astype(np.uint8))
            save_image(inpainted_image_pil, f"rgb_pano{file_suffix}", offset + pos, self.args.rgb_path)
            
            save_image(depth_pil, f"depth{file_suffix}", offset + pos, self.args.depth_path)
            #save_image(normal_pil, f"normal{file_suffix}", offset + pos, self.args.depth_path)
            #save_image(depth_pil_not_aligned, f"depth_not_aligned{file_suffix}", offset + pos, self.args.depth_path)
            #save_rgbd(inpainted_image_pil, depth_pil, f'rgbd{file_suffix}', offset + pos, self.args.rgbd_path)
            
        # remove masked-out faces. If we use erosion in the mask it means those points will be removed.
        # it may cause holes if the scene is complex
        if self.args.replace_over_inpainted:
            # only now update mask: predicted depth will still take old positions as anchors (they are still somewhat correct)
            # otherwise if we erode/dilate too much we could get depth estimates that are way off
            if not self.args.update_mask_after_improvement:
                self.inpaint_mask = self.inpaint_mask + self.eroded_dilated_inpaint_mask

            self.remove_masked_out_faces()

        # add new points (novel content)
        inpaint_mask_equi = torch.ones((H_pano, W_pano), device=self.args.device, dtype=self.inpaint_mask.dtype)
        inpaint_mask_equi[256:768, :] = self.inpaint_mask
        inpainted_threshold_mask_cubes = equi2cube(equi=inpaint_mask_equi.unsqueeze(0).to(torch.float32), rots={"roll":0, "pitch":0, "yaw":0}, w_face=512, cube_format="list")
        inpainted_image_cubes = equi2cube(equi=inpainted_image_equi.squeeze(0), rots={"roll":0, "pitch":0, "yaw":0}, w_face=512, cube_format="list")
        
        aligned_depth_equi = torch.zeros((H_pano, W_pano), device=self.args.device, dtype=self.rendered_depth.dtype)
        aligned_depth_equi[256:768, :] = aligned_depth
        aligned_depth = aligned_depth_equi
        depth_max = aligned_depth.max()
        if depth_max > 50:
            depth_max = 50
            aligned_depth = torch.clamp(aligned_depth, max=depth_max)
        aligned_depth_cubes = equi2cube(equi=(aligned_depth/depth_max).unsqueeze(0), rots={"roll":0, "pitch":0, "yaw":0}, w_face=512, cube_format="list")

        for i, world_to_cam in enumerate(self.cubemap_world_to_cams):

            self.world_to_cam = world_to_cam
            original_fov = self.args.fov
            self.args.fov = 90
            _, _, _ = self.project()
            self.inpaint_mask = (inpaint_masks_cubes[i][0, ...].to(torch.float32) * inpainted_threshold_mask_cubes[i][0, ...])
            self.inpaint_mask[self.inpaint_mask < 0.9] = 0
            self.inpaint_mask = self.inpaint_mask.to(torch.bool)
            #print(f"rgb: {inpainted_image_cubes[i].shape}, depth: {(aligned_depth_cubes[i]*depth_max / depth_to_distance_ratio(512, 512, self.K)).squeeze(0).shape}, mask: {self.inpaint_mask.shape}")
            final_depth = (aligned_depth_cubes[i]*depth_max / depth_to_distance_ratio(512, 512, self.K)).squeeze(0)
            self.inpaint_mask[final_depth == 0] = 0
            self.add_vertices_and_faces(inpainted_image_cubes[i], final_depth, use_opencv_camera_with_intrinsic)
            self.args.fov = original_fov


        # save current meshes
        if save_files and self.args.save_files and self.args.save_scene_every_nth > 0 and (offset + pos) % self.args.save_scene_every_nth == 0:
            self.save_mesh(f"fused_until_frame{file_suffix}_{offset + pos:04}.ply")

        return inpainted_image_pil, inpainted_image, predicted_depth, predicted_norm, aligned_depth
    
