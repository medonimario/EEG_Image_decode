<div align="center">

<h2 style="border-bottom: 1px solid lightgray;">Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion</h2>

<!-- Badges and Links Section -->
<div style="display: flex; align-items: center; justify-content: center;">
    <a href="https://arxiv.org/pdf/2403.07721" style="margin-right: 10px;">
        <img alt="Paper" src="http://img.shields.io/badge/Paper-arxiv.2404.07202-B31B1B.svg">
    </a>
    <a href="https://huggingface.co/datasets/LidongYang/EEG_Image_decode" style="margin-right: 10px;">
        <img alt="Hugging Face" src="https://img.shields.io/badge/EEG_Image_decode-%F0%9F%A4%97%20Hugging%20Face-blue">
    </a>
</div>

<br/>

<!-- Images Section -->
<!-- <div align="center">
    <img src="docs/images/teaser.png" width="100%"/>
</div> -->

<!-- <div align="center">
    <img src="docs/images/demo_example.png" width="100%"/>
</div> -->


</div>


<!-- 
<img src="bs=16_test_acc.png" alt="Framework" style="max-width: 90%; height: auto;"/> -->
<!-- 
<img src="test_acc.png" alt="Framework" style="max-width: 90%; height: auto;"/> -->

<!-- As the training epochs increases, the test set accuracy of different methods. (Top: batchsize is 16. Bottom: batchsize is 1024) -->

<!-- 
<img src="temporal_analysis.png" alt="Framework" style="max-width: 90%; height: auto;"/>
Examples of growing window image reconstruction with 5 different random seeds. -->


<img src="framework.png" alt="Framework" style="max-width: 100%; height: auto;"/>
Framework of the proposed method.




<!--  -->
<img src="fig-genexample.png" alt="fig-genexample" style="max-width: 90%; height: auto;"/>  

Some examples of using EEG to reconstruct stimulus images.





## Environment setup
Run ``setup.sh`` to quickly create a conda environment that contains the packages necessary to run our scripts; activate the environment with conda activate BCI.
```
. setup.sh
```
You can also create a new conda environment and install the required dependencies by running
```
conda env create -f environment.yml
conda activate BCI

pip install wandb
pip install einops
```
Additional environments needed to run all the code:
```
pip install open_clip_torch

pip install transformers==4.28.0.dev0
pip install diffusers==0.24.0

#Below are the braindecode installation commands for the most common use cases.
pip install braindecode==0.8.1
```
## Quick training and test 

If you want to quickly reproduce the results in the paper, please download the relevant ``preprocessed data`` and ``model weights`` from [Hugging Face](https://huggingface.co/datasets/LidongYang/EEG_Image_decode) first.
#### 1.Image Retrieval
We provide the script to learn the training strategy of EEG Encoder and verify it during training. Please modify your data set path and run:
```
cd Retrieval/
python ATMS_retrieval.py --logger True --gpu cuda:0  --output_dir ./outputs/contrast
```
We also provide the script for ``joint subject training``, which aims to train all subjects jointly and test on a specific subject:
```
cd Retrieval/
python ATMS_retrieval_joint_train.py --joint_train --sub sub-01 True --logger True --gpu cuda:0  --output_dir ./outputs/contrast
```
#### 2.Image Reconstruction
We provide scripts for image reconstruction. Please modify your data set path and run zero-shot on 200 classes test dataset:
```
# step 1: reconstruct images
cd Generation/
Generation_metrics_sub<index>.ipynb

# step 2: compute metrics
cd Generation/fMRI-reconstruction-NSD/src
Reconstruction_Metrics_ATM.ipynb
```
Also, We also provide scripts for image reconstruction combined ``with the low level pipeline``.
```
# step 1: train vae encoder and then generate low level images
cd Generation/
train_vae_latent_512_low_level_no_average.py

# step 2: load low level images and then reconstruct them
cd Generation/
1x1024_reconstruct_sdxl.ipynb
```

## Data availability
We provide you with the ``preprocessed EEG`` and ``preprocessed MEG`` data used in our paper at [Hugging Face](https://huggingface.co/datasets/LidongYang/EEG_Image_decode), as well as the raw image data.


Note that the experimental paradigms of the THINGS-EEG and THINGS-MEG datasets themselves are different, so we will provide images and data for the two datasets separately.

You can also download the relevant THINGS-EEG data set and THINGS-MEG data set at osf.io.

The raw and preprocessed EEG dataset, the training and test images are available on [osf](https://osf.io/3jk45/).
- ``Raw EEG data:`` `../project_directory/eeg_dataset/raw_data/`.
- ``Preprocessed EEG data:`` `../project_directory/eeg_dataset/preprocessed_data/`.
- ``Training and test images:`` `../project_directory/image_set/`.


The raw and preprocessed MEG dataset, the training and test images are available on [OpenNEURO](https://openneuro.org/datasets/ds004212/versions/2.0.0).





## EEG/MEG preprocessing
Modify your path and execute the following code to perform the same preprocessing on the raw data as in our experiment:
```
cd EEG-preprocessing/
python EEG-preprocessing/preprocessing.py
```

```
cd MEG-preprocessing/
MEG-preprocessing/pre_possess.ipynb
```
Also You can get the data set used in this project through the BaiduNetDisk [link](https://pan.baidu.com/s/1-1hgpoi4nereLVqE4ylE_g?pwd=nid5) to run the code.

## Acknowledge

1.Thanks to Y Song et al. for their contribution in data set preprocessing and neural network structure, we refer to their work:</br>"[Decoding Natural Images from EEG for Object Recognition](https://arxiv.org/pdf/2308.13234.pdf)".</br> Yonghao Song, Bingchuan Liu, Xiang Li, Nanlin Shi, Yijun Wang, and Xiaorong Gao. 

2.We also thank the authors of [SDRecon](https://github.com/yu-takagi/StableDiffusionReconstruction) for providing the codes and the results. Some parts of the training script are based on [MindEye](https://medarc-ai.github.io/mindeye/). Thanks for the awesome research works.

3.Here we provide our THING-EEG dataset cited in the paper:</br>"[A large and rich EEG dataset for modeling human visual object recognition](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)".</br>
Alessandro T. Gifford, Kshitij Dwivedi, Gemma Roig, Radoslaw M. Cichy.


4.Another used THINGS-MEG data set provides a reference:</br>"[THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior.](https://elifesciences.org/articles/82580.pdf)".</br> Hebart, Martin N., Oliver Contier, Lina Teichmann, Adam H. Rockter, Charles Y. Zheng, Alexis Kidder, Anna Corriveau, Maryam Vaziri-Pashkam, and Chris I. Baker.

## Citation

```bibtex
@article{li2024visual,
  title={Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion},
  author={Li, Dongyang and Wei, Chen and Li, Shiying and Zou, Jiachen and Liu, Quanying},
  journal={arXiv preprint arXiv:2403.07721},
  year={2024}
}
```