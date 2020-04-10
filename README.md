# CFMNet
 Flexible Image Denoising with Multi-layer Conditional Feature Modulation

## Prerequisites:
* Python 3.6
* PyTorch 1.0
* NVIDIA GPU + CUDA cuDNN 10.0

##Introduction:
For flexible non-blind image denoising, existing deep networks usually take both noisy image and noise level map as the input to handle various noise levels with a single model. However, in this kind of solution, the noise variance (i.e., noise level) is only deployed to modulate the first layer of convolution feature with channel-wise shifting, which is limited in balancing noise removal and detail preservation. In this paper, we present a novel flexible image denoising network (CFMNet) by equipping a U-Net backbone with multi-layer conditional feature modulation (CFM) modules. 
In comparison to channel-wise shifting only in the first layer, CFMNet can make better use of noise level information by deploying multiple layers of CFM. Moreover, each CFM module takes convolutional features from both noisy image and noise level map as input for better trade-off between noise removal and detail preservation. 

The four figures are the denoising results of image 148026 from the CBSD68 dataset with noise standard deviation 60 by FFDNet and our CFMNet. Incomparison, our CFMNet (σin= 60) achieves better trade-off between noise removal and detail preservation.
It can be seen that FFDNet with the input noise level 60 is effective in removing noise, but may smooth out some small-scale details (see Fig (a)).In comparison, FFDNet with the input noise level 55, i.e., FFDNet (sigma in=55), can preserve more details but some noise is still retained in the result (see Fig.(b)).
[a](https://github.com/dujiazhi/CFMNet/blob/master/figures/noisy60.png)
[b](https://github.com/dujiazhi/CFMNet/blob/master/figures/ffdnet_5560.png) 
[c](https://github.com/dujiazhi/CFMNet/blob/master/figures/ffdnet_6060.png) 
[d](https://github.com/dujiazhi/CFMNet/blob/master/figures/CFMNet60.png) 
Experimental results show that our CFMNet is effective in exploiting noise level information for flexible non-blind denoising, and performs favorably against the existing deep image denoising methods in terms of both quantitative metrics and visual quality.
Detailed description of the system can be found in our paper.

## Network Architecture:
![architecture]( https://github.com/dujiazhi/CFMNet/blob/master/figures/CFMNet.png)  
## Test FFDNet Models
[test.py]( https://github.com/dujiazhi/CFMNet/blob/master/test.py) is the testing demo of CFMNet for denoising images corrupted by AWGN.
[train.py]( https://github.com/dujiazhi/CFMNet/blob/master/train.py) is the training demo of CFMNet for denoising images corrupted by AWGN.

### Image Denoising for AWGN:
>### Grayscale Image Denoising:
![d](https://github.com/dujiazhi/CFMNet/blob/master/figures/gray.png) 

>### Color Image Denoising
![d](https://github.com/dujiazhi/CFMNet/blob/master/figures/rgb.png) 





