# GenRC: 3D Indoor Scene Generation from Sparse Image Collections
### [Project Page](https://minfenli.github.io/GenRC/) 
<!-- | [arXiv](https://arxiv.org/abs/) -->

This repository will contain the official implementation of [GenRC](https://minfenli.github.io/GenRC/), an automated training-free pipeline to complete a room-scale 3D mesh with sparse RGBD observations. The source codes will be released soon.

![Teaser](https://minfenli.github.io/GenRC/images/pipeline.png "GenRC")

**Pipeline of GenRC:** (a) Firstly, we extract text embeddings as a token to represent the style of provided RGBD images via textual inversion. Next, we project these images to a 3D mesh. (b) Following that, we render a panorama from a plausible room center and use equirectangular projection to render various viewpoints of the scene from the panoramic image. Then, we propose E-Diffusion that satisfies equirectangular geometry to concurrently denoise these images and determine their depth via monocular depth estimation, resulting in a cross-view consistent panoramic RGBD image. (c) Lastly, we sample novel views from the mesh to fill in holes, resulting in a complete mesh.

## BibTeX
```
@inproceedings{minfenli2024GenRC,
  title={GenRC: 3D Indoor Scene Generation from Sparse Image Collections},
  author={Ming-Feng Li, Yueh-Feng Ku, Hong-Xuan Yen, Chi Liu, Yu-Lun Liu, Albert Y. C. Chen, Cheng-Hao Kuo, Min Sun},
  booktitle=ECCV,
  year={2024}
}
```
