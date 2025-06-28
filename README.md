# CycleGAN-Picture-to-Van-Gogh

Image to image translation using CycleGAN: Picture to Van Gogh

The image-to-image translation is a type of computer vision that aims to learn the correspondence between an input and an output image through training image pairs. Each pair comprises image A belonging to domain X and its corresponding image B in domain Y.

Paired training data can be applied in a supervised learning manner, but paired images may only be available for some tasks. To address this, **CycleGAN** is an approach that enables the learning of image translation from a source domain X to a target domain Y without paired images.
This project aims to train a model that transforms an actual image into a **Van Gogh**-style image as depicted in the given image.

To learn more about **CycleGAN**, we recommend referring to the original paper at: https://arxiv.org/abs/1703.10593.

The dataset required for this project can be found here: [vangogh2photo.zip] https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/
