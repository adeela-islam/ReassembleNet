# ReassembleNet: Learnable Keypoints and Diffusion for 2D Fresco Reconstruction

This repository contains the official implementation of our paper:
[ICCV], 2025  
[Paper link (arXiv/DOI)](https://arxiv.org/pdf/2505.21117)

---

## 🧩 Overview
The task of reassembly is a significant challenge across multiple domains, including archaeology, genomics, and molecular docking, requiring the precise placement and orientation of elements to reconstruct an original structure. In this work, we address key limitations in state-of-the-art Deep Learning methods for reassembly, namely i) scalability; ii) multimodality; and iii) real-world applicability: beyond square or simple geometric shapes, realistic and complex erosion, or other real-world problems. We propose ReassembleNet, a method that reduces complexity by representing each input piece as a set of contour keypoints and learning to select the most informative ones by Graph Neural Networks pooling inspired techniques. ReassembleNet effectively lowers computational complexity while enabling the integration of features from multiple modalities, including both geometric and texture data. Further enhanced through pretraining on a semi-synthetic dataset. We then apply diffusion-based pose estimation to recover the original structure.

<p align="center">
  <img src="https://github.com/adeela-islam/ReassembleNet/blob/main/docs/method.png" width="1000"/>
</p>

### Installation

```bash
pip install -r requirements.txt
pip install -e .

### 📊 Dataset Preparation
```bash
cd scripts
