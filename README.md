# End-to-End Planning for Autonomous Driving
### CIVIL-459 Deep Learning for Autonomous Vehicles  — EPFL, Spring 2026
**Tomas Garate Anderegg · Jules Streit**

---

## Overview

This repository contains our implementation of an end-to-end deep learning planner for autonomous driving, developed as part of the CIVIL-459 Deep Learning for Autonomous Vehicles course at EPFL. The goal is to predict future vehicle trajectories directly from raw sensor inputs, without relying on a hand-engineered modular pipeline.

We use a curated subset of the **nuPlan** dataset, the world's first large-scale planning benchmark for autonomous vehicles. The model takes as input camera images, driving commands, vehicle motion history and outputs a sequence of future waypoints evaluated via **Average Displacement Error (ADE)**.

The project is structured into **three progressive milestones**, each available on its own branch:

| Branch | Milestone | Key addition |
|---|---|---|
| `milestone-1` | Basic End-to-End Planner | Camera + driving command + history → trajectory |
| `milestone-2` | Perception-Aware Planning | + Auxiliary tasks: semantic segmentation & depth estimation |
| `milestone-3` | Sim-to-Real Generalization | + Domain adaptation & data augmentation for out-of-distribution robustness |

---

## Dataset

Each sample is a Python dictionary with the following fields:

```
camera              (200, 300, 3)   # RGB image at timestep 21
depth               (200, 300, 1)   # Depth image at timestep 21
driving_command     str             # 'forward' | 'left' | 'right'
sdc_history_feature (21,  3)       # [x, y, heading] over the past 21 steps
sdc_future_feature  (60,  3)       # [x, y, heading] for the next 60 steps (target)
semantic_label      (200, 300)      # Semantic segmentation map
```

All coordinates are expressed in the ego vehicle's local frame at timestep 21. The training set contains 5,000 samples, the validation set 1,000.

---

## Milestones

### Milestone 1 — Basic End-to-End Planner (`Milestone_1`)

**Allowed inputs:** `camera`, `driving_command`, `sdc_history_feature`

We implement a baseline end-to-end neural network that fuses visual and motion features to predict the 60-step future trajectory.

**Our approach:**
- A **CNN backbone** (ResNet-based) encodes the RGB image into a spatial feature map
- The **driving command** is embedded and concatenated as a conditioning signal
- The **motion history** (21 × 3) is encoded via a GRU
- All features are fused and decoded into a 60-step trajectory via an MLP head

---

### Milestone 2 — Perception-Aware Planning (`Milestone_2`)

**Additional inputs:** `semantic_label`, `depth` (as auxiliary supervision only)

We extend the Milestone 1 model with multi-task learning to improve the richness of learned visual representations.

**Our approach:**
- Two **auxiliary decoder heads** are added to the shared backbone:
  - Semantic segmentation head (predicts the 15-class semantic map)
  - Depth estimation head (predicts the per-pixel depth map)
- The backbone is jointly trained with a **weighted multi-task loss** combining trajectory regression, segmentation cross-entropy, and depth L1 loss
- Auxiliary heads are discarded at inference time; only the trajectory head is used

---

### Milestone 3 — Sim-to-Real Generalization (`Milestone_3`)

**Note:** No depth or semantic labels are available in this phase.

We evaluate the planner on out-of-distribution, real-world-like domains and focus on robustness through data augmentation and domain adaptation strategies.

**Our approach:**
- Strong **photometric and geometric augmentations** (color jitter, random crop, horizontal flip, Gaussian blur) to reduce reliance on simulation-specific visual cues
- Optional **feature-space alignment** to encourage domain-invariant representations
- The model from Milestone 2 serves as the starting point, with fine-tuning on the augmented training distribution

---

## Setup

1. Create a new conda environment with Python 3.11:

```bash
conda create -n dlav_env python=3.11
conda activate dlav_env
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Main dependencies:

```
torch>=2.0.0
torchvision>=0.15.0
numpy
matplotlib
gdown
wandb
tqdm
Pillow
```

## Compute

Training is run on the **EPFL SCITAS Izar cluster** (GPU nodes). Example SLURM submission scripts are provided in `configs/slurm/`.

---

## Authors

- **Tomas Garate Anderegg** — EPFL MA4
- **Jules Streit** — EPFL MA4

Course: CS-503 Visual Intelligence, Spring 2026
