<p align="center"><img width=25% src="https://github.com/aangelopoulos/conformal-classification/blob/master/media/logo_conformal_compat.svg"></p>
<p align="center"><img width=50% src="https://github.com/aangelopoulos/conformal-classification/blob/master/media/text_conformal.svg"></p>

<p align="center">
    <a href="" alt="Python"> <img src="https://img.shields.io/badge/python-v3.6+-blue.svg" /> </a>
    <a href="" alt="Dependencies"> <img src="https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg" /> </a>
    <a href="https://opensource.org/licenses/MIT" alt="License"> <img src="https://img.shields.io/badge/license-MIT-blue.svg" /> </a>
</p>

## Paper 
[Uncertainty Sets for Image Classifiers using Conformal Prediction](https://arxiv.org/abs/)
```
@article{angelopoulos2020event,
  title={Uncertainty Sets for Image Classifiers using Conformal Prediction},
  author={Angelopoulos, Anastasios N and Bates, Stephen and Malik, Jitendra and Jordan, Michael I},
  journal={arXiv preprint arXiv:},
  year={2020}
}
```

## Basic Overview

<p>
    This codebase modifies any PyTorch classifier to output a <i>predictive set</i> which provably contains the true class with a probability you specify.
    It uses a method called Regularized Adaptive Prediction Sets (RAPS), which we introduce in our accompanying paper.
    The procedure is as simple and fast as Platt scaling, but provides a formal guarantee for every model and dataset.
</p>

<figure>
<img src="https://github.com/aangelopoulos/conformal-classification/blob/master/media/figure_sets.svg" alt="Set-valued classifier." style="display: block; width=80%">
<figcaption>
    <b>Prediction set examples on Imagenet.</b> we show three examples of the class <tt>fox squirrel</tt> along with 95% prediction sets generated by our method to illustrate how set size changes based on the difficulty of a test-time image.
</figcaption>
</figure>

<br>

## Usage
From the root directory, install the dependencies and run our example by executing:
```
git clone https://github.com/aangelopoulos/conformal-classification
cd conformal-classification
conda env create -f environment.yml
conda activate conformal
python example.py 'path/to/imagenet/val/'
```
If you'd like to use our codebase on your own model, first place this at the top of your file:
```
from conformal.py import *
from utils.py import *
```
Then create a holdout set for conformal calibration using a line like: 

[`calib, val = random_split(mydataset, [num_calib,total-num_calib])` ](https://github.com/aangelopoulos/conformal-classification/blob/b3823a924bbd039b60bf5a37e517ca87f598fdbe/example.py#L39)

Finally, you can choose `kreg` and `lamda` and conformalize your model with, e.g.,

[`model = ConformalModel(model, calib_loader, alpha=0.1, kreg=5, lamda=0.01)`](https://github.com/aangelopoulos/conformal-classification/blob/b3823a924bbd039b60bf5a37e517ca87f598fdbe/example.py#L53)

## Expected outputs
The output of `example.py` with `seed=0` and `num_calib=2000` should be:
```
Computing logits for model (only happens once).
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:33<00:00,  2.06s/it]
Optimal T=1.2172279357910156
Model calibrated and conformalized! Now evaluate over remaining data.
N: 48000 | Time: 2.318 (2.401) | Loss: 0.7886 (0.8778) | Cvg@1: 0.773 (0.783) | Cvg@5: 0.945 (0.940) | Cvg@RAPS: 0.914 (0.902) | Size@RAPS: 4.242 (4.012)
Complete!
```
The values in parentheses are running averages. The preceding values are only for the most recent batch. The timing values will be different on your system, but the rest of the numbers should be exactly the same. 

The expected outputs of the experiments are stored in `experiments/outputs`, and they are exactly identical to the results reported in our paper. You can reproduce the results by executing `python table1.py`, `python figure2.py`, and `python figure4.py` after you have installed our dependencies.

## Picking `alpha`, `kreg`, and `lamda`

`alpha` is the maximum proportion of errors you are willing to tolerate. The target coverage is therefore `1-alpha`. A smaller `alpha` will usually lead to larger sets, since the desired coverage is more stringent.

`kreg` is the first class at which the RAPS penalty is applied. `kreg` should ideally be `1+kfixed`, where `kfixed` is the smallest fixed-size set at which your classifier achieves the coverage guarantee you want. In practice we have found it suffices for many models to pick `kreg=5`, but performance can be improved by optimizing `kreg`. The specific choice of `kreg` matters less for small values of `lamda`.

`lamda` is the level of RAPS regularization. It is a nonnegative real number. Any `lamda` above 1 is equivalent. The larger `lamda` is, the more the RAPS sets shrink towards a fixed set size. We purposefully misspell lambda as `lamda` because of the conflicting Python keyword.

## License
<a href="https://opensource.org/licenses/MIT" alt="License">MIT License</a>
