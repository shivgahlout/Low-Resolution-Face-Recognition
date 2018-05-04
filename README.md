# Low-Resolution-Face-Recognition


The procedure for the experiments is as follows:

1.The LWF dataset has 13223 images belonging to 5749 classes. Out of these 4069 classes have single image only. 

2. We consider only those images that have at least 10 images. It gives use 159 classes. 

3. We split these images into 85$\%$ train, 5$\%$ val and 15$\%$ test dataset. 

4. After this we perform alignment of the images. 

5. To generate the low resolution images, we first downsample with scale s and then upscale using bicubic interpolation using same scale. 

## Architecture

1. The architecture has into two parts:  First we use a pretrained VGG architecture. This architecture is trained on 2.6M images in 2.6K categories. We fine this architecture for our dataset. ii) We use CNN for super resolution of the images. We use SRCNN for this purpose. We train this model on low resolution images. We minimize the PSNR to train this architecture. Finally, we combine both of these architecture and fine tune on LR image. 

## Results

1. The accuracy obtained on HR images is 95.2%
2. The accuracy obtained on LR images is 95.1%
5. The accuracy obtained on LR (by a factor of 3) images is 89.75%
