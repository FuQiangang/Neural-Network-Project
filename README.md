# Neural-Network-Project

## The Problem 
The goal of this exposition is to state the Open Set Recognition problem rigorously and to discuss possible solutions to tackle it. 

Let $\left(\Omega,\mathcal{F}, ℙ \right)$ be the underlying probability space and let partition $\tilde{\mathcal{P}}$ over $ℝ^{d}\times \\{-1,1\\}$ be a class. Then such a partition $\tilde{\mathcal{P}}$ induces a function $l_{\tilde{P}}: ℝ^{d}\to \\{-1,1\\}$. Thus, for any $x\in ℝ^{d}$, $\left(x, l_{\tilde{P}}(x) \right)\in \mathcal{P}$. Now, let $V:\Omega\to ℝ^{d}$ be a random vector, and thus we can define a random variable $L:= l_{\tilde{P}}(V): \left(\Omega,\mathcal{F}, ℙ \right)\to \\{-1,1 \\}$. This $L$ is called the label of $V$. If $L = 1$, we say $V$ belongs to the class to be recognized. If $L = -1$, $V$ is from any other class.  

Now, let $\hat{V} = \\{v_1,...,v_m\\}$ be the tranining samples drawn from $P$. That is, each $v_i \in \hat{V}$ is a realization of independent copy of $V$ and $\left(v_i,1\right)\in\tilde{P}$. Moreover, let $\hat{K} = \\{k_1,...,k_n \\}$ from other known classes $K\subset \tilde{P}$. That is, each $k_i \in \hat{K}$ is also a relization of independent copy of $V$ such that $(k_i,-1)\in \tilde{P}$. Furthermore, let $U$ be a subset of unknown classes, that is, $P\cup K\cup U$ is a disjoint union and $P\cup K\cup U \subset \tilde{P}$. Finally, let test data $\\{t_1,...,t_z \\}\in P\cup K\cup U$. Then, we first observe that the test data contains unknown classes and is disjoint from the training set. 

In this project, we used our trained GAN on one data set containing facial images to test if this GAN can accurately identify fake or real images in a completely different images, which come from modern era. However, the images we used to train come from 1990s.  


# Part 2: Data
Training data set: https://www.bioid.com/face-database/
Validation data and Testing data: Byron provided to me in our shared folder. 
Another 5 fake testing data come from https://this-person-does-not-exist.com/en

## Description of the Dataset
For training data, we used the BioID Face Database that contains 1,521 grayscale images. All samples appear to be high-density samples of specific individuals. Each subject appears in multiple images under varying conditions. The validation data set contains 1974 images of regular human colorful facial images. For the fake testing data set, there are 1988 fake human facial images. 

The major difference between training and validation data sets is the following. The training data sets is a limited number of specific subjects (biometric focus) with multiple samples per person to teach the model specific identity features. However, the validation Set originates from the FFHQ dataset, which has different lighting, higher resolution origins, and much higher subject diversity. It is important to make sure the validation data set is very different than the training data set. I used two sources of testing data. The first one comes from Byron, and contains 1988 images. The second one is from https://this-person-does-not-exist.com/en. I tested 5 of them. 

After data processing, they are all exactly greyscale $64 \times 64$ pixels and all image-based. Due to greyscale, we aim to concentrate on spatial factors rather than color information. 

Instead of splitting the original dataset, I used a completely separate dataset for validation. This provides a stronger evaluation of generalization, since the model is tested on data drawn from a different distribution. The observed drop in performance indicates that the discriminator has partially overfitted to dataset-specific features rather than learning a fully distribution-invariant representation of faces.

# Part 3: Challenges:
## Lack of evaluation metric: 
One of the challenges encountered was the absence of a well-defined evaluation metric for the generative model. Unlike supervised learning tasks, where accuracy or loss directly measures performance, GANs do not provide a straightforward notion of correctness. In our current implementation, the discriminator outputs values: D(x)∈[0,1]. The output of the discriminator represents the probability that an image is real. During training, we tracked the following. First, D(x): discriminator score on real images. Second, D(G(z)): discriminator score on generated images.

However, these values are not reliable indicators of model quality. First, a high D(x) may simply indicate overfitting to training data and a low D(G(z)) does not necessarily imply poor generation quality. The discriminator may become too strong or too weak, distorting these signals. Thus, evaluation relied heavily on visual inspection, which is subjective and difficult to quantify. 

## Preprocessing Constraints: 
All images were resized to 64 × 64 grayscale, which simplifies the problem but introduces limitations. For instance, loss of fine facial details, reduced texture diversity, and potential distortion from resizing. Additionally, grayscale conversion removes color information, which could be useful for distinguishing features.

## Training Instability
GAN training is inherently unstable due to its minimax optimization structure, minmax L(G,D). However, this creates a dynamic where the discriminator and generator compete
Improvements in one can destabilize the other. In my practice, this led to oscillating loss values, sudden degradation in generated image quality, and sensitivity to hyperparameters. For example, when the discriminator becomes too strong, the generator receives vanishing gradients. On the other hand, if the generator becomes too strong, the discriminator fails to learn. 

## Absence of a Proper Validation Method
Another major challenge was the lack of a structured validation procedure. Initially, the code only evaluated discriminator outputs during training using the same dataset: $X\sim \mu_{ref}$. This creates a misleading picture of performance, since the model is evaluated on the same data it was trained on. A key difficulty was determining how to construct a validation process for a GAN. Unlike standard models, GANs do not naturally separate training and validation objectives. We considered two approaches, splitting the dataset into training and validation subsets and using a completely separate dataset. When I applied the first method, 

The second approach was ultimately more appealing, since it evaluates generalization under distribution shift. However, this introduces additional complications, such as ensuring compatibility between datasets (resolution, grayscale format, normalization). Implementing this correctly required careful alignment of data preprocessing. As the result showed below, in fact, this big distribution shift 

# Part 4: 
## This is the training code: [Colab notebook](https://colab.research.google.com/drive/1q4tmwzm65zC3ltJuRGpJmStOP39vwdlQ?usp=sharing)

## This is the testing code: [Colab notebook](https://colab.research.google.com/drive/1OC2PGweVT6bePwWyCmZF5KjT3dmaRuSw?usp=sharing)
There are two data sources for testing part. The first source comes from fake images Byron provided. In the code, they are denoted as 
We ran AUC and got the following result: <img width="959" height="721" alt="Screenshot 2026-04-29 at 11 36 57 PM" src="https://github.com/user-attachments/assets/4f2506b8-171a-439d-b8ba-c32eedff9d42" />

The 0.41 AUC score shows that the Discriminator has learned the wrong rules. It consistently gives higher Realness scores to the Fakes than to the new Real test images. This shows GNA doesn't work well in an open class recognition problem.  

We also showed softmax probability distribution as follow 

<img width="1129" height="623" alt="Screenshot 2026-04-30 at 10 01 21 AM" src="https://github.com/user-attachments/assets/327f1fc0-d236-4124-b2f1-93ef83daefdf" />

This picture also verifies GNA model's failure to generalize. First, The Real (Blue) Peak is sitting almost exactly at 0.500. In a perfect model, this should be over at 1.0. The Fake pictures from "Face Doesn't Exist" (Green) Peak is actually sitting slightly to the right of the Real peak (around 0.501). Lastly, The Fake (Red) Peak is the most spread out, with a significant tail stretching toward 0.504. This reaffirms that the model thinks the Fakes look more "Real" than the actual Real images. 

Another observation is of high density. This means the Discriminator is giving almost every real image the exact same score. The tight clustering of all scores between 0.498 and 0.504 shows a model that has stiffened. The weights are so concentrated that the model can no longer move the probability significantly away from 0.5.


# Part 5:
## Problem A:  Overfitting to "Sensor Noise" and Legacy Format
The BioID training set was captured with late-90s monochrome CCD cameras, which have a specific grain and low dynamic range. My discriminator has reached 100% accuracy by memorizing these specific pixel patterns rather than general facial geometry. When testing on modern, higher-resolution CMOS sensor images (Real Test Data), the cleaner digital signal looks like a "Fake" to the model.

##  Problem B: The Aspect Ratio and Padding Trap
As seen in my test visualization, the CenterCrop transformation often leaves black padding bars on the top or bottom of images that don't have a 1:1 aspect ratio. What went wrong, probably is the training set did not have these black bars. The Discriminator uses these high-contrast edges as a shortcut to classify the image as OOD (Fake), regardless of whether there is a face in the center.

## Potential Improvement A: Advanced Data Augmentation
Instead of a simple CenterCrop, I will implement a more robust pipeline to break the model's reliance on specific pixel locations. Random Resized Crop: This forces the model to recognize faces at different scales and positions, eliminating the "black bar" bias.

On the other hand, Adding synthetic grain to the modern test images can help bridge the gap between modern sensors and the legacy BioID sensor style.

## Potential Improvement B: Label Smoothing.
Train the model using $0.9$ as the target for Real instead of $1.0$. This perhaps prevents the model from becoming overconfident in its narrow training distribution.
