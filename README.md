# Attention Block U-Net
# Improved Photoacoustic Imaging of Numerical Bone Model Based on Attention Block U-Net Deep Learning Network

Source code for ''[Improved Photoacoustic Imaging of Numerical Bone Model Based on Attention Block U-Net Deep Learning Network](https://www.mdpi.com/2076-3417/10/22/8089)''.

# Models

We designed an Attention Block U-Net (AB U-Net) Network from the standard U-Net by integrating the attention blocks in the feature extraction part, aiming to be more adaptive for imaging bone samples with complex structure.

![image](https://github.com/Panpan-Chen/Attention-Block-U-net/blob/main/images/AB_UNet.png)

The attention blocks originated from [Convolutional block attention module](https://arxiv.org/abs/1807.06521) (CBAM). 	

![](https://github.com/Panpan-Chen/Attention-Block-U-net/blob/main/images/Attention_Block.png)

[U-Net](https://arxiv.org/abs/1505.04597) and [Attention U-Net](https://arxiv.org/abs/1804.03999) models are also contained in this repository.

# Train

`python main.py`

# Test

Run test.py

# Visualization

The curves of BCE loss, PSNR and SSIM with iterations are realized by TensorBoard.

`# runs
tensorboard --logdir runs`

# Results

## Visual comparison of the performance in three examples based on Time Reversal and AB U-Net

AB U-Net successfully removes artifacts and restores the high-frequency information, such as the micro-structure of the trabecular bone. Compared with Time Reversal method, the CNN-based network provides significant improvement in PSNR and SSIM, i.e., SSIM of sample 1 increases from 0.62 to 0.88, indicating an accurate modeling of the initial pressure.

![](https://github.com/Panpan-Chen/Attention-Block-U-net/blob/main/images/performance.png)
