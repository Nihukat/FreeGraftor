<div align="center">

<h1>FreeGraftor</h1>
<h3>FreeGraftor: Training-Free Cross-Image Feature Grafting for Subject-Driven Text-to-Image Generation</h3>

Zebin Yao, &nbsp; Lei Ren, &nbsp; Huixing Jiang, &nbsp; Wei Chen, &nbsp; Xiaojie Wang, &nbsp; Ruifan Li, &nbsp; Fangxiang Feng

[![arXiv](https://img.shields.io/badge/arXiv-<2504.15958>-<COLOR>.svg)](https://arxiv.org/abs/2504.15958)

</div>

## 🛠️ Installation

```bash
git clone https://github.com/Nihukat/FreeGraftor.git
cd FreeGraftor
conda create -n freegraftor python==3.10.9
conda activate freegraftor
pip install -r requirements.txt
```

## 📝 Preparation

### 1. Download SAM checkpoints.

```bash
aria2c -x 4 -j 5 https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 
```

### 2. Download FLUX.1 and Grounding DINO from huggingface.

FLUX.1:
```bash
huggingface-cli download --token hf_your_huggingface_token --resume-download black-forest-labs/FLUX.1-dev --local-dir your/directory/for/FLUX.1-dev
```

Grounding DINO:
```bash
huggingface-cli download --resume-download IDEA-Research/grounding-dino-tiny --local-dir your/directory/for/grounding-dino-tiny
```

### 3. Setup model paths

Set the paths to the downloaded models in your environment variables:
```bash
export FLUX_DEV=your/directory/for/FLUX.1-dev
export GROUNDING_DINO=your/directory/for/grounding-dino-tiny
export SAM=./sam_vit_h_4b8939.pth
```

You can also uncomment and modify the paths in the `src/generate.py` and `src/app.py` files:
```python
os.environ['FLUX_DEV'] = 'your/directory/for/FLUX.1-dev'
os.environ['GROUNDING_DINO'] = 'your/directory/for/grounding-dino-tiny'
os.environ['SAM'] = './sam_vit_h_4b8939.pth'
```

## 🚀 Usage

### Basic Usage

**Single-concept examples :**

```bash
python src/generate.py \
--gen_prompt "A can is placed on a desk, next to a laptop." \
--class_names "beer can" \
--image_paths "examples/can_00.jpg"\
--output_dir "inference_results" \
--start_seed 0 \
--num_images 2 \

```

You can also use a configuration file to pass parameters of the concept:

```bash
python src/generate.py \
--gen_prompt "A can is placed on a desk, next to a laptop." \
--concept_config_path "configs/can.json"
--output_dir "inference_results" \
--start_seed 0 \
--num_images 2 \

```

**Multi-subject examples :**

```bash
python src/generate.py \
--gen_prompt "A cat and a dog in Times Square, zoom in, close-up. " \
--class_names "cat" "dog" \
--image_paths "examples/cat_01.jpg" "examples/dog_02.jpg" \
--output_dir "inference_results" \
--start_seed 0 \
--num_images 2 \

```

You can also use a configuration file to pass parameters of the concepts:

```bash
python src/generate.py \
--gen_prompt "A cat and a dog in Times Square, zoom in, close-up. " \
--concept_config_path "configs/cat_dog.json"
--output_dir "inference_results" \
--start_seed 0 \
--num_images 2 \

```

### Gradio Demo

```bash
python src/app.py
```

## ✅ To-Do List

- [ ] More Usage and Applications
- [ ] Support for Stable Diffusion 3
- [ ] Visualization Results
- [x] Gradio Demo
- [x] Source Code

## 💖 Acknowledgements
We sincerely thank [FireFlow](https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing), [RF-Solver](https://github.com/wangjiangshan0725/RF-Solver-Edit) and [FLUX](https://github.com/black-forest-labs/flux/tree/main) for their well-structured codebases. The support and contributions of the open-source community have been invaluable, and without their efforts, completing our work so efficiently would not have been possible. 

## 📚 Citation

If you find this code useful for your research, please consider citing:

```
@article{yao2025freegraftor,
  title={FreeGraftor: Training-Free Cross-Image Feature Grafting for Subject-Driven Text-to-Image Generation},
  author={Yao, Zebin and Ren, Lei and Jiang, Huixing and Wei, Chen and Wang, Xiaojie and Li, Ruifan and Feng, Fangxiang},
  journal={arXiv preprint arXiv:2504.15958},
  year={2025}
}
```
