import torch
import math
import os
import cv2
from PIL import Image
import numpy as np
import json
from tqdm.auto import tqdm


from model.utils.utils import (
    save_image,
    pil_to_torch,
    load_sd_inpaint_one_step,
    erode_and_dilate_mask
)

from equilib import cube2equi, equi2pers, equi2cube
from model.PerspectiveAndEquirectangular.lib import Equirec2Perspec as E2P
from model.PerspectiveAndEquirectangular.lib import multi_Perspec2Equirec as m_P2E
from torchvision import transforms




class DiffusionPipeline(torch.nn.Module):
    @torch.no_grad()
    def __init__(self, args, setup_models=True, H=512, W=512):
        super().__init__()
        # setup (create out_dir, save args)
        self.args = args

        # load models if required
        if setup_models:
            self.setup_models_one_step()


    @torch.no_grad()
    def setup_models_one_step(self):
        # construct inpainting stable diffusion pipeline
        self.inpaint_pipe = load_sd_inpaint_one_step(self.args)


    @torch.no_grad()
    def remove_models(self):
        self.inpaint_pipe = None
        torch.cuda.empty_cache()
    

    @torch.no_grad()
    def inpaint_one_step(self, rendered_image_pil, inpaint_mask_pil, latent, diffusion_step, num_inference_steps):
        latent = latent.unsqueeze(0)    # for batch dimention
        latent, single_step_latent = self.inpaint_pipe.inpaint_one_step(
            prompt=self.args.prompt,
            negative_prompt=self.args.negative_prompt,
            image=rendered_image_pil,
            mask_image=inpaint_mask_pil,
            guidance_scale=self.args.guidance_scale_panorama,
            num_inference_steps=num_inference_steps,
            latents = latent,
            i = diffusion_step,
        )

        return latent[0], single_step_latent[0]


    def get_views_cord(self, H, W, window_size=64, stride=16):
        '''
        returns starts and ends for MultiDiffusion thing
        '''
        H /= 8
        W /= 8
        num_blocks_height = (H - window_size) // stride + 1
        num_blocks_width = (W) // stride + 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size
            views.append((h_start, h_end, w_start, w_end))
        return views[:-1]
    

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


    @torch.no_grad()
    def generate_panorama_MultiDiffusion(self, rendered_images, inpaint_masks, strength=1.0, pano_latent=None):
        '''
        generate panorama using MultiDiffusion
        '''
        to_pil = transforms.ToPILImage()
        H_pano = 512
        W_pano = 512*4
        H_pano_latent = 64
        W_pano_latent = 64*4

        rendered_image_pil_list = []
        inpaint_mask_pil_list = []


        for i, (h_start, h_end, w_start, w_end) in enumerate(self.get_views_cord(H=H_pano, W=W_pano)):
            if w_end <= W_pano_latent:
                rendered_image_pil_list.append(to_pil(rendered_images[:, :, w_start*8:w_end*8]))
                inpaint_mask_pil_list.append(to_pil(erode_and_dilate_mask(inpaint_masks[:, :, w_start*8:w_end*8])))

            else:
                rendered_image_pil_list.append(to_pil(torch.cat((rendered_images[:, :, w_start*8:], rendered_images[:, :, :w_end*8-W_pano]), dim=2)))
                inpaint_mask_pil_list.append(to_pil(erode_and_dilate_mask(torch.cat((inpaint_masks[:, :, w_start*8:], inpaint_masks[:, :, :w_end*8-W_pano]), dim=2))))



        self.inpaint_pipe.scheduler.set_timesteps(self.args.panorama_num_inference_steps, device=self.args.device)
        timesteps, num_inference_steps = self.inpaint_pipe.get_timesteps(
            num_inference_steps=self.args.panorama_num_inference_steps, strength=strength, device=self.args.device)
        
        noise = torch.randn((4, H_pano_latent, W_pano_latent), device=self.args.device, dtype=torch.float16)
        if strength != 1:
            pano_latent = self.inpaint_pipe._encode_vae_image((pano_latent*2-1).clamp(-1, 1).unsqueeze(0).to(torch.float16), generator=None).squeeze(0)
            averaged_latents = self.inpaint_pipe.scheduler.add_noise(pano_latent, noise, timesteps[:1])
        else:
            averaged_latents = noise
            
        for diffusion_step, t in enumerate(tqdm(timesteps, desc="Texture refinement w/ MultiDiffusion")):
            denoised_latent_value = torch.zeros_like(averaged_latents)
            denoised_latent_count = torch.zeros_like(averaged_latents)

            for i, (h_start, h_end, w_start, w_end) in enumerate(self.get_views_cord(H=H_pano, W=W_pano)):
                rendered_image_pil = rendered_image_pil_list[i]
                inpaint_mask_pil = inpaint_mask_pil_list[i]
                
                if w_end <= W_pano_latent:
                    averaged_latent = averaged_latents[:, :, w_start:w_end]
                else:
                    averaged_latent =torch.cat((averaged_latents[:, :, w_start:], averaged_latents[:, :, :w_end-W_pano_latent]), dim=2)

                inpaint_output = self.inpaint_one_step(rendered_image_pil, inpaint_mask_pil, averaged_latent, diffusion_step + self.args.panorama_num_inference_steps - len(timesteps), self.args.panorama_num_inference_steps)

                if w_end <= W_pano_latent:
                    denoised_latent_value[:, :, w_start:w_end] += inpaint_output[0]
                    denoised_latent_count[:, :, w_start:w_end] += 1
                else:
                    denoised_latent_value[:, :, w_start:] += inpaint_output[0][:, :, :W_pano_latent-w_start]
                    denoised_latent_value[:, :, :w_end-W_pano_latent] += inpaint_output[0][:, :, W_pano_latent-w_start:]
                    denoised_latent_count[:, :, w_start:] += 1
                    denoised_latent_count[:, :, :w_end-W_pano_latent] += 1
                
            averaged_latents = torch.where(denoised_latent_count > 0, denoised_latent_value / denoised_latent_count, denoised_latent_value)

        
        inpainted_image_pil = self.inpaint_pipe.vae.decode(averaged_latents.unsqueeze(0) / self.inpaint_pipe.vae.config.scaling_factor, return_dict=False)[0]
        inpainted_image_pil = self.inpaint_pipe.image_processor.postprocess(inpainted_image_pil, output_type="pil", do_denormalize=[True] * inpainted_image_pil.shape[0])[0]
        inpainted_image = pil_to_torch(inpainted_image_pil, device = self.args.device)
        save_image(inpainted_image_pil, f"after_texture_refinement", 0, self.args.out_path)

        return inpainted_image
    

    @torch.no_grad()
    def generate_panorama_E_Diffusion(self, rendered_image_equi, inpaint_mask_equi, num_views=8, cutoff=1.0):
        to_pil = transforms.ToPILImage()
        H_pano = 512*2
        W_pano = 512*4
        H_pano_latent = 64*2
        W_pano_latent = 64*4

        views = []
        fovs = []
        for i in range(num_views):
            views.append({'roll': 0, 'pitch': 0, 'yaw': -np.pi/num_views*2*i})
            fovs.append(98)
        
        m_P2E_views = []

        denoised_latent_equi = torch.zeros((4, 64*2, 64*4), device = self.args.device)

        single_step_latents = [None for i in views]
        noise = []

        rendered_image_pers_list = []
        inpaint_mask_pers_list = []

        for i, view in enumerate(views):
            m_P2E_views.append([fovs[i], -view['yaw']/np.pi*180, -view['pitch']/np.pi*180])

            rendered_image_pers_list.append(equi2pers(equi=rendered_image_equi, rots=view, height=512, width=512, fov_x=fovs[i]))
            
            inpaint_mask_pers = equi2pers(equi=inpaint_mask_equi.to(torch.float32), rots=view, height=512, width=512, fov_x=fovs[i])
            inpaint_mask_pers = erode_and_dilate_mask(inpaint_mask_pers)
            inpaint_mask_pers_list.append(inpaint_mask_pers)
        
        save_image(to_pil(torch.from_numpy(np.concatenate([i.cpu().numpy() for i in rendered_image_pers_list], axis=2)).to(device=self.args.device)), f"rendered_perspectives", 0, self.args.out_path)
        save_image(to_pil(torch.from_numpy(np.concatenate([i.cpu().numpy() for i in inpaint_mask_pers_list], axis=2)).to(device=self.args.device)), f"mask_perspectives", 0, self.args.out_path)
        
        
        

        for diffusion_step in tqdm(range(self.args.panorama_num_inference_steps), desc="EG-MultiDiffusion"):
            if diffusion_step >= cutoff * self.args.panorama_num_inference_steps:
                break            
            
            for i, view in enumerate(views):
                rendered_image_pers = rendered_image_pers_list[i]
                inpaint_mask_pers = inpaint_mask_pers_list[i]
                
                
                if diffusion_step == 0:
                    # initialize x_{T} as Gaussian noise
                    noise.append(torch.randn((4, 64, 64), device=self.args.device, dtype=torch.float16))
                    averaged_latent = noise[i]
                else:
                    # warping latents to i-th view
                    ones = np.transpose(np.ones_like(single_step_latents[i]), (1, 2, 0))
                    latent_pers_value = np.transpose(np.zeros_like(single_step_latents[i]), (1, 2, 0))
                    latnet_pers_count = np.transpose(np.zeros_like(single_step_latents[i]), (1, 2, 0))
                    for j, _ in enumerate(views):
                        if self.radians_difference_less_than_90_deg(views[i]['yaw'], views[j]['yaw']) and self.radians_difference_less_than_90_deg(views[i]['pitch'], views[j]['pitch']):
                            K,      R      = self.get_K_R(fovs[j], views[j]['yaw']/np.pi*180, -views[j]['pitch']/np.pi*180, 64, 64)
                            self_K, self_R = self.get_K_R(fovs[i], views[i]['yaw']/np.pi*180, -views[i]['pitch']/np.pi*180, 64, 64)
                            H = self_K @ self_R @ R.T @ np.linalg.inv(K)
                            latent_pers_value += cv2.warpPerspective(np.transpose(single_step_latents[j], (1, 2, 0)), H, (64, 64), flags=cv2.INTER_CUBIC)
                            latnet_pers_count += cv2.warpPerspective(ones, H, (64, 64), flags=cv2.INTER_CUBIC)
                            
                    denoised_latent_pers = torch.tensor(np.where(latnet_pers_count, latent_pers_value / latnet_pers_count, 0), device=self.args.device).permute(2, 0, 1)
                    
                    averaged_latent = denoised_latent_pers.to(torch.float16)
                    
                    # sample random noise every two steps
                    if diffusion_step % 2 == 0:
                        noise[i] = torch.randn((4, 64, 64), device=self.args.device, dtype=torch.float16)
                                        
                    # add noise
                    timesteps, _ = self.inpaint_pipe.get_timesteps(num_inference_steps=self.args.panorama_num_inference_steps, strength=1, device=self.args.device)
                    timestep = int(timesteps[diffusion_step])
                    averaged_latent = averaged_latent * self.inpaint_pipe.scheduler.alpha_t[timestep] + noise[i] * torch.sqrt(1 - self.inpaint_pipe.scheduler.alphas_cumprod[timestep])

                # denoise to step t-1
                inpaint_output = self.inpaint_one_step(to_pil(rendered_image_pers), to_pil(inpaint_mask_pers), averaged_latent, diffusion_step, self.args.panorama_num_inference_steps)
                
                single_step_latents[i] = inpaint_output[1].to(torch.float32).cpu().numpy()              # predicted x_0 in latent space
             
            
            '''
            per = m_P2E.Perspective(single_step_latents, m_P2E_views)
            denoised_image_equi = torch.from_numpy(per.GetEquirec(H_pano_latent, W_pano_latent, 4)[:, 32:96, :]).to(device=self.args.device)
            inpainted_image_pil = self.inpaint_pipe.vae.decode(denoised_image_equi.to(torch.float16).unsqueeze(0) / self.inpaint_pipe.vae.config.scaling_factor, return_dict=False)[0]
            inpainted_image_pil = self.inpaint_pipe.image_processor.postprocess(inpainted_image_pil, output_type="pil", do_denormalize=[True] * single_step_latents[i].shape[0])[0]
            save_image(inpainted_image_pil, f"EGMD_step", diffusion_step, self.args.out_path)
            #'''
        
        denoised_images = [None for i in views]
        for i, view in enumerate(views):
            denoised_images[i] = self.inpaint_pipe.vae.decode(torch.tensor(single_step_latents[i], dtype=torch.float16, device=self.args.device).unsqueeze(0) / self.inpaint_pipe.vae.config.scaling_factor, return_dict=False)[0].squeeze(0).to(torch.float32).cpu().numpy() / 2 + 0.5
        
        per = m_P2E.Perspective(denoised_images, m_P2E_views)
        denoised_image_equi = torch.from_numpy(per.GetEquirec(H_pano, W_pano, 3, self.args.panorama_stitching_alpha_blending_at_border_width)).to(device=self.args.device)    
        denoised_image_equi = torch.clamp(denoised_image_equi, min=0, max=1)
        inpainted_image = denoised_image_equi[:, 256:768, :]
        save_image(to_pil(torch.from_numpy(np.concatenate(denoised_images, axis=2)).to(device=self.args.device)), f"after_EGMD_perspectives", 0, self.args.out_path)
        save_image(to_pil(inpainted_image), f"after_EGMD", 0, self.args.out_path)

        return inpainted_image


    
    @torch.no_grad()
    def inpaint_panorama(self, image_cubemap=None, inpaint_mask_cubemap=None, prompt=None, negative_prompt=None):
        H_pano = 512 * 2
        W_pano = 512 * 4

        to_pil = transforms.ToPILImage()

        if prompt is not None:
            self.args.prompt = prompt
        if negative_prompt is not None:
            self.args.negative_prompt = negative_prompt
        

        if image_cubemap is not None and inpaint_mask_cubemap is not None:
            # inpaint small seams
            for i in range(len(image_cubemap)):
                m = inpaint_mask_cubemap[i][0, ...].cpu().numpy().astype(np.uint8)
                m[m!=0] = -1
                image_cubemap[i] = rendered_images_equi = torch.tensor(cv2.inpaint(np.asarray(to_pil(image_cubemap[i])), m, 3, cv2.INPAINT_TELEA), device=image_cubemap[i].device, dtype=image_cubemap[i].dtype).permute(2, 0, 1) / 255

            rendered_images_equi = cube2equi(cubemap=image_cubemap, cube_format="list", height=H_pano, width=W_pano)
            inpaint_masks_equi = cube2equi(cubemap=inpaint_mask_cubemap, cube_format="list", height=H_pano, width=W_pano)
        else:
            rendered_images_equi = torch.zeros(3, H_pano, W_pano)
            inpaint_masks_equi = torch.ones(3, H_pano, W_pano)

        rendered_images = rendered_images_equi[:, 256:768, :]
        inpaint_masks = inpaint_masks_equi[:, 256:768, :]


        save_image(to_pil(rendered_images), f"input_rgb_pano", 0, self.args.out_path)
        save_image(to_pil(inpaint_masks), f"input_mask_pano", 0, self.args.out_path)


        hybrid_split=self.args.panorama_diffusion_hybrid_split    # this amount of warp and (1-this) amount of normal MD
        inpainted_images = self.generate_panorama_E_Diffusion(rendered_images_equi, inpaint_masks_equi, num_views=8, cutoff=hybrid_split)
        inpainted_images = self.generate_panorama_MultiDiffusion(rendered_images, inpaint_masks, strength=1-hybrid_split, pano_latent=inpainted_images)

        
        #save_image(to_pil(inpainted_images), f"output_rgb_pano", 0, self.args.out_path)
    
        return inpainted_images
    

    def inpaint_perspective_image(self, **kwargs):
        return self.inpaint_pipe(**kwargs)
