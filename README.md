# Semantic-CariGANs
Code for "Learning to Caricature via Semantic Shape Transformation"

## Authors
Wenqing Chu, Wei-Chih Hung, Yi-Hsuan Tsai, Yu-Ting Chang, Yijun Li, Deng Cai, Ming-Hsuan Yang

## Dataset
- [Webcaricature](https://cs.nju.edu.cn/rl/WebCaricature.htm): contains the photos and caricatures.

## Dependency
- PyTorch >= 0.4.1
- Python = 3.6

## Usage
Our method has four parts, segmentation, retrieval, shape transformation and style transfer.
We provide the pretrained segmentation, retrieval, style adaptation and shape adaptation models in Google Drive (https://drive.google.com/open?id=1x0bJ7wBAsC_jSjm30SIqjl-huuBZMKh6)
Please put the models under 'Semantic-CariGANs/'.
For testing options, please use an aligned photo image and run the below commond. Refer to 'options/test_options.py' for testing setting and details.
```
python predict.py --name parseref_gan
```
The training code will be here soon, stay tuned.


## Reference
If you use the code or our dataset, please cite our paper

@article{Chu2020Learning,
    title={Learning to Caricature via Semantic Shape Transformation},
    author={Chu, Wenqing and Hung, Wei-Chih and Tsai, Yi-Hsuan Chang, Yu-Ting and Li, Yijun and Cai, Deng and Yang, Ming-Hsuan},
    year={2020},
}


## Acknowledgment
This code is heavily borrowed from
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [CycleGan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN)
