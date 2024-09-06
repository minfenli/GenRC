# GenRC: 3D Indoor Scene Generation from Sparse Image Collections
### [Project Page](https://minfenli.github.io/GenRC/) | [arXiv](https://arxiv.org/abs/2407.12939)

This repository contains the official implementation of [GenRC](https://minfenli.github.io/GenRC/), an automated training-free pipeline to complete a room-scale 3D mesh with sparse RGBD observations. The source codes will be released soon.

![Teaser](https://minfenli.github.io/GenRC/images/pipeline.png "GenRC")

**Pipeline of GenRC:** (a) Firstly, we extract text embeddings as a token to represent the style of provided RGBD images via textual inversion. Next, we project these images to a 3D mesh. (b) Following that, we render a panorama from a plausible room center and use equirectangular projection to render various viewpoints of the scene from the panoramic image. Then, we propose E-Diffusion that satisfies equirectangular geometry to concurrently denoise these images and determine their depth via monocular depth estimation, resulting in a cross-view consistent panoramic RGBD image. (c) Lastly, we sample novel views from the mesh to fill in holes, resulting in a complete mesh.

## Prepare Environment

Create a conda environment:

```bash
conda create -n genrc python=3.9 -y
conda activate genrc
pip install -r requirements.txt
```

Then install Pytorch3D by following the [official instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
For example, to install Pytorch3D on Linux (tested with PyTorch 1.13.1, CUDA 11.7, Pytorch3D 0.7.2):

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

Download the pretrained model weights for the fixed depth inpainting model, that we use:

- refer to the [official IronDepth implemention](https://github.com/baegwangbin/IronDepth) to download the files ```normal_scannet.pt``` and ```irondepth_scannet.pt```.
- place the files under ```genrc/checkpoints```

(Optional) Download the pretrained model weights for the text-to-image model:

```bash
git clone https://huggingface.co/stabilityai/stable-diffusion-2-inpainting
git clone https://huggingface.co/stabilityai/stable-diffusion-2-1
ln -s <path/to/stable-diffusion-2-inpainting> checkpoints
ln -s <path/to/stable-diffusion-2-1> checkpoints
```

(Optional) Download the repository and install packages for textual inversion:
```
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
cd examples/textual_inversion
pip install -r requirements.txt
```

## Prepare Dataset

Download [the preprocessed ScanNetV2 dataset](https://drive.google.com/file/d/12MUFPsLxJakr5bnLO5XsyGQ4lEN9q2Wb/view?usp=share_link). Extract via:

```bash
mkdir input && unzip scans_keyframe.zip -d input && mv input/scans_keyframe input/ScanNetV2
```

(Optional) Prepare test frames for textual inversion and prepare a script file for running textual inversion.
```bash
python prepare_textual_inversion_data_RGBD2.py 
python prepare_textual_inversion_script_RGBD2.py 

# run textual inversion
source textual_inversion.sh
```

## Inference
To run and evaluate on ScanNet dataset:

```
python generate_scene_scannet_rgbd2_test_all.py --exp_name ScanNetV2_exp
```

To run the baseline method, T2R+RGBD: (please refer to our paper)

```
python generate_scene_scannet_rgbd2_test_all.py --exp_name ScanNetV2_exp --method "T2R+RGBD"
```


## Evaluate
(Optional) Generate ground-truth mesh and calculate one-directional Chamfer distance:

```
python calculate_chamfer_completeness.py --exp_name ScanNetV2_exp
```

Merge all calculated metrics into one csv file:

```
python merge_test_metrics.py --exp_name ScanNetV2_exp
```

## BibTeX
```
@inproceedings{minfenli2024GenRC,
  title={GenRC: 3D Indoor Scene Generation from Sparse Image Collections},
  author={Ming-Feng Li, Yueh-Feng Ku, Hong-Xuan Yen, Chi Liu, Yu-Lun Liu, Albert Y. C. Chen, Cheng-Hao Kuo, Min Sun},
  booktitle=ECCV,
  year={2024}
}
```


## Acknowledgements

Thanks to the following projects for providing their open-source codebases and models.

- [StableDiffusion](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)
- [IronDepth](https://github.com/baegwangbin/IronDepth)
- [Text2Room](https://github.com/lukasHoel/text2room)