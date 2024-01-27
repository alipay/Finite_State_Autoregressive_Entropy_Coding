# Finite-State Autoregressive Entropy Coding for Efficient Learned Lossless Compression

This is the official Pytorch implementation of our ICLR paper:

[Finite-State Autoregressive Entropy Coding for Efficient Learned Lossless Compression](https://openreview.net/forum?id=D5mJSNtUtv&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions))

## Setup

### Hardware Requirements
* CUDA compatible GPU (Optional but highly recommended for training. For testing, no GPU is OK)
* Memory >= 32G (Optional but required for training on ImageNet64. We force full dataset caching for ImageNet64, which requires ~12G memory per dataset instance.)


### Software Requirements
* Linux/Mac system (Windows may be possible but not tested)
* gcc&g++>=7 (Required, need c++17 support to compile)
* python>=3.7 (Recommended version, need pytorch-lightning support)
* pytorch>=1.7 (Recommended version, need pytorch-lightning support)
* cudatoolkit>=10.2 (Optional but highly recommended, lower versions may be functional)
* [craystack](https://github.com/j-towns/craystack) (Optional, if you want to compare with BB-ANS)

The recommended enrironment setup script with conda: 
```bash
conda create -n cbench python=3.7
conda activate cbench
conda install -c pytorch pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2
# conda install -c pytorch pytorch==1.12.1 torchvision==0.13.1 # cudatoolkit=11.3
# if gcc version < 7
conda install -c conda-forge gcc gxx
pip install -r requirements.txt
python setup.py build develop
```

### (Optional) Environment setup according to your machine
See configs/env.py

### (Optional) 3rdparty setup
```bash
git submodules update --init --recursive
cd 3rdparty/craystack
python setup.py build develop
```

## Dataset Prepare
We use 5 datasets in our experiments:
* CIFAR10 : No need for preparation, torchvision would automatically download for you.
* [ImageNet32/64](https://clic.compression.cc/) : Download downsampled images 32x32 and 64x64 and decompress them into data/ImageNet. The result folders would be:
  * data/ImageNet/Imagenet32_train_npz
  * data/ImageNet/Imagenet32_val_npz
  * data/ImageNet/Imagenet64_train_npz
  * data/ImageNet/Imagenet64_val_npz
* [OpenImages](https://storage.googleapis.com/openimages/web/download_v6.html) : We follow [L3C](https://github.com/fab-jul/L3C-PyTorch?tab=readme-ov-file#prepare-open-images-for-training) to preprocess OpenImages as training/validation set. The result folders would be:
  * data/openimages/clean_train
  * data/openimages/clean_val
* [Kodak](https://r0k.us/graphics/kodak/) : Download all images to data/Kodak
* [CLIC 2020 validation](https://data.vision.ee.ethz.ch/cvl/clic/professional_valid_2020.zip) : Download and unzip all images to data/CLIC/valid

## Code Structure
* configs : All parameter setting configs
* cbench : All algorithm codes
  * benchmark : Tools for experiments and comparisons
  * csrc : c/c++ codes, with an efficient ANS-based entropy coder interface/implementation. (Should be more efficient than CompressAI implementation, we use pybind11 Tensor support to avoid memory format conversion between python and c++.)
  * codecs : All compression codecs
  * data : Tools for data loading and processing
  * modules : Mainly compression modules
    * prior_model/prior_coder/latent.py : Regarding the proposed FSAR, please refer to CategoricalAutoregressivePriorDistributionPriorCoder; For STHQ, please refer to StochasticVQAutoregressivePriorDistributionPriorCoder.
  * nn : Neural network codes.
  * utils : Misc tools.
* tests : To test if some modules are functional.
* tools : Tools to run experiments.
  * run_benchmark.py : Use this for all experiments!

## Experiments
For any experiment you want to run (including Training/Validation/Testing, thanks to pytorch-lightning and our BasicLosslessCompressionBenchmark):
```bash
python tools/run_benchmark.py [config_file]
```

You can use tensorboard to visualize the training process.

### Experiment List
NOTE: If an GPU out-of-memory error occured, adjust batch_size_total in each config file. In our experiments, we use 8 A100 for most configs so the batch_size_total might be too large.
* configs/coding_exp_cf10.py (and im32, im64, oi) : Main codec comparison experiments. Could test compression ratio and speed for different image codecs.
* configs/model_shallow_exp_cifar10.py (and im32, im64) : Main latent model comparison experiments. Could test compression ratio and speed for different latent models.
* configs/model_shallow_exp_abl_cifar10.py : Ablation study for different VQ methods.
* configs/model_abl_bb_exp_cifar10.py : Ablation study for different backbone setting.
* configs/dsar_exp.py : Ablation study for data/observation space autoregressive coding.

### Model Implementation List
See configs/presets.

Our full model is tagged as "V2DVQ-c2-FSAR-O2S-catreduce1.5"

### Pretrained Models
TBA

tools/run_benchmark.py can automatically look for config.pth in a given directory to build the benchmark. Therefore, to test a pretrained model, simply run:
```bash
python tools/run_benchmark.py [model_directory]
```

## Citation
TBA

## Contact
TBA