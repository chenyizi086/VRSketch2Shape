<div align="center">

# Order Matters: 3D Shape Generation from Sequential VR Sketches

</div>

<h4 align="center">
Yizi Chen*, Sidi Wu*, Tianyi Xiao, Nina Wiedemann, Loic Landrieu
</h4>

## Description
Pytorch implementation of the paper [Order Matters: 3D Shape Generation from Sequential VR Sketches](https://chenyizi086.github.io/VRSketch2Shape_website/)

We introduce Sketch2Shape model:
- An automated pipeline that generates sequential VR sketches from arbitrary shapes
- A dataset of over 20k synthetic and 900 hand-drawn sketch-shape pairs across four categories
- An order-aware sketch encoder coupled with a diffusion-based 3D generator

<p align="center">
<img src="media/pipeline.png" width="800"/>
</p>

### Project Structure

Structure of this repository:

```
|
├── dataloader                   <- Data loader
├── config                       <- Model configurations
├── eval                         <- Evaluation code  
├── data                         <- Dataset for training
│   ├── VRSketch2Shape           <- VRSketch2Shape dataset
├── models                       <- Model
│   ├── base_model.py            <- Base model
|   ├── sketch2shape_model.py    <- VRSketch2Shape model
├── environment.yml              <- Conda environment .yml file
├── utils                        <- Some useful functions
├── infer.py                     <- Inference and evaluation code for VRSketch2Shape
└── README.md
```

## Installation

### 1. Create and activate conda environment
```
conda env create -f environment.yml
conda activate sketch2shape
```

### 2. Download datasets from huggingface repo
At the moment, we only provide sketch shapes for model inference. The training sketch shapes will be released soon!
```
pip install huggingface_hub
huggingface-cli download YiziChen/VRSketch2Shape_dataset/blob/main/data.zip --local-dir .
unzip data.zip
```

### 3. Download weights from huggingface repo
```
pip install huggingface_hub
huggingface-cli download YiziChen/sketch2model/df_epoch_best_multicls.pth --local-dir ./weights/all_class
```

## How to run

### Training the model

Coming soon!

### Testing the model

To test and evaluate the model, launch:
```bash
sh scripts/run_infer.sh
```

### Qualitative results


### Citation   

If you use this method in your work, please cite our [paper](https://arxiv.org/pdf/2512.04761).

```markdown
@inproceedings{Chen2025OrderM3,
  title={Order Matters: 3D Shape Generation from Sequential VR Sketches},
  author={Yizi Chen and Sidi Wu and Tianyi Xiao and Nina Wiedemann and Loic Landrieu},
  doi={10.48550/arXiv.2512.04761},
  year={2025},
}
```

## Acknowledgement
We are thankful for the great open-source code of [SDFusion](https://github.com/yccyenchicheng/SDFusion).

## Issues and FAQ

Coming soon!