# Scene Graph Generation for Semantic Image Understanding

A deep learning pipeline for generating scene graphs from images, enabling semantic understanding of visual relationships between objects.


## Overview
This project leverages **Detectron2** and **Visual Genome** dataset to generate scene graphs that represent objects and their relationships in an image. The system detects objects, classifies them, and infers semantic relationships between them.

## Features
- Object detection using Faster R-CNN with ResNet-101 backbone
- Pre-trained on Visual Genome dataset
- Scene graph generation with object-relationship-object triplets
- Visualization of detected objects and relationships
- Customizable threshold for object and relationship detection

## Installation
1. Clone the repository:
```bash
git clone https://github.com/mariam-gad232/graph-scene-generation.git
```

## Install dependencies:

```
pip install torch torchvision detectron2 opencv-python matplotlib
```
