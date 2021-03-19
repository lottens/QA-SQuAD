# QANet and BiDAF: Reading Comprehension Default Final Project

## Introduction

QANet implementation in PyTorch. The original paper for QANet can be found here: https://arxiv.org/pdf/1804.09541.pdf.

## QANet Usage
Run `conda activate squad`

Debug by CDing into qanet and running `python QANet.py --batch_size 16 --epochs 3 --with_cuda --use_ema --debug`

Train by CDing into qanet and running `python QANet.py --batch_size 16 --epochs 30 --with_cuda --use_ema`

Test by CDing into squad and running the test script for the dev or test split `python test.py --split dev --load_path ../qanet/save/model_best.pth.tar --name dev_submission.csv`

## BiDAF Usage
See course-provided README in /squad

## Credits
All setup code was modeled after BangLiu's implementation at: github.com/BangLiu/QANet-PyTorch. Modifications were made to the model architecture, to adopt tensorboard, to support unanswerable questions in Squad2.0 and compute new metrics like AvNA.
