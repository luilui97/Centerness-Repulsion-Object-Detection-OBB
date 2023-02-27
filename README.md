This repository is the source code for paper:

# **Center-ness and Repulsion: Constraints to Improve Remote Sensing Object Detection via RepPoints**

## Introduction

Remote sensing object detection is a basic yet challenging task in remote sensing image understanding. In contrast to horizontal objects, remote sensing objects are commonly densely packed with arbitrary orientations and highly complex backgrounds. Existing object detection methods lack an effective mechanism to exploit these characteristics and distinguish various targets. Unlike mainstream approaches ignoring spatial interaction among targets, this paper proposes a shape-adaptive repulsion constraint on point representation to capture geometric information of densely distributed remote sensing objects with arbitrary orientations. Specifically, (1) we first introduce a shape-adaptive center-ness quality assessment strategy to penalize the bounding boxes having a large margin shift from the center point. Then, (2) we design a novel oriented repulsion regression loss to distinguish densely packed targets: closer to the target and farther from surrounding objects. Experimental results on four challenging datasets including DOTA, HRSC2016, UCAS-AOD, and WHU-RSONE-OBB, demonstrate the effectiveness of our proposed approach.

## Installation

Please see [README_mmrotate](README_mmrotate.md) to install MMRotate

## File 

[config files](configs/repulsion_centerness_reppoints/) is available to train our proposed model.

The head of our model is availabel at [models/dense_heads/repulsion_oriented_reppoints_head.py](models/dense_heads/repulsion_oriented_reppoints_head.py)

Repulsion loss can be found at [Repulsion Loss is available at models/losses/repulsion_reppoints_loss.py](Repulsion Loss is available at models/losses/repulsion_reppoints_loss.py)