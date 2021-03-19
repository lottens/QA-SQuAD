# Reading Comprehension Default Final Project

## Introduction

QANet implementation in PyTorch. The original paper for QANet can be found here: https://arxiv.org/pdf/1804.09541.pdf.

## Usage
Run `conda activate squad`

Train baseline approaches by CDing into squad and following second level README

Debug main approaches by CDing into qanet and running `python QANet.py --batch_size 16 --epochs 3 --with_cuda --use_ema --debug`

Train main approaches by CDing into qanet and running `python QANet.py --batch_size 16 --epochs 30 --with_cuda --use_ema`

Test by CDing into squad and running the test script for the dev or test split `python test.py --split dev --load_path ../qanet/save/model_best.pth.tar --name dev_submission.csv`

## Credits
All setup code for QANet was modeled after BangLiu's implementation at: github.com/BangLiu/QANet-PyTorch. Modifications were made to the model architecture, to adopt tensorboard, to support unanswerable questions in Squad2.0 and compute new metrics like AvNA.
