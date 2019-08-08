# Conditional GAN based Synthetic Oversampler

Over-sampling method to generate synthetic observations for specific classes based on the Pytorch implementation of a conditional GAN. The conditional GAN is trained on the Wasserstein loss including a gradient penalty (WGAN-GP) or Fisher IPM loss.
The oversampler generates categorical variables by generating their softmax output and 1) using their weighted-averaged embeddings during critic training and 2) sampling with softmax probabilities from a categorical distribution during sampling of the synthetic data.

## Usage
There is a nice wrapper to fit the GAN to a numpy array of data and an integration into the imbalance-learn/scikit-learn framework. See jupyter notebook GANbalancer_Tutorial for an introduction. 

## Sources and inspiration

* https://github.com/jalola/improved-wgan-pytorch
* https://github.com/caogang/wgan-gp
* https://github.com/kuc2477/pytorch-wgan-gp

* Mottini, A., Lheritier, A., & Acuna-Agost, R. (2018). Airline Passenger Name Record Generation using Generative Adversarial Networks. ArXiv e-prints. https://ui.adsabs.harvard.edu/#abs/2018arXiv180706657M
