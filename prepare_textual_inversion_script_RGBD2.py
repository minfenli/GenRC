import os
import shutil
from PIL import Image
import numpy as np
# from glob import glob

from dataset.scannet_rgbd2 import ScanNetRGBD2

def main():
    data_path = 'input/ScanNetV2'
    
    with open(os.path.join(data_path, "test_scenes.txt"),'r') as fp:
        test_scenes = fp.readlines()
        test_scenes = [x.strip('\n') for x in test_scenes]
    print(test_scenes)

    outdir = 'input/ScanNetV2_test'

    device = 0

    sample_percents = [5, 10, 20, 50]

    with open('textual_inversion.sh', 'w') as fp:

        for scene_name in test_scenes:
            scene_outdir = os.path.join(outdir, scene_name)

            for percent in sample_percents:
                views_outdir = os.path.join(scene_outdir, str(percent))

                fp.write(("" if device is None else f"CUDA_VISIBLE_DEVICES={device} ") +
                    'accelerate launch textual_inversion.py --pretrained_model_name_or_path="checkpoints/stable-diffusion-2-1"' + 
                    f" --train_data_dir=\"{views_outdir}\"" + 
                    ' --learnable_property="full" --placeholder_token="<style>" --initializer_token="room" --resolution=512' +
                    ' --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=3000 --learning_rate=5.0e-04 --checkpointing_steps=10000' + 
                    ' --scale_lr --lr_scheduler="constant" --lr_warmup_steps=0 --num_vectors=5' +
                    f" --output_dir=\"{os.path.join(scene_outdir, 'textual_inversion', str(percent))}\"" +
                    '\n')

if __name__ == "__main__":
    main()
