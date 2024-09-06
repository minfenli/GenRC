import os
import shutil
from PIL import Image
import numpy as np
from glob import glob

from dataset.scannet_rgbd2 import ScanNetRGBD2

def main():
    data_path = 'input/ScanNetV2'
    
    with open(os.path.join(data_path, "test_scenes.txt"),'r') as fp:
        test_scenes = fp.readlines()
        test_scenes = [x.strip('\n') for x in test_scenes]
    print(test_scenes)

    outdir = 'input/ScanNetV2_test'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    sample_percents = [5, 10, 20, 50]

    for scene_name in test_scenes:
        scene_outdir = os.path.join(outdir, scene_name)
        if not os.path.exists(scene_outdir):
            os.mkdir(scene_outdir)

        for percent in sample_percents:
            views_outdir = os.path.join(scene_outdir, str(percent))
            if not os.path.exists(views_outdir):
                os.mkdir(views_outdir)
                
            log_dirs = glob(os.path.join(scene_outdir, 'textual_inversion', '*', 'logs'))
            for log_dir in log_dirs:
                shutil.rmtree(log_dir)


if __name__ == "__main__":
    main()
