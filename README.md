# Neural-Network-Project

## The Problem 
The goal of this exposition is to state the Open Set Recognition problem rigorously and to discuss possible solutions to tackle it. 

Let $\left(\Omega,\mathcal{F}, ℙ \right)$ be the underlying probability space and let partition $\tilde{\mathcal{P}}$ over $ℝ^{d}\times \\{-1,1\\}$ be a class. Then such a partition $\tilde{\mathcal{P}}$ induces a function $l_{\tilde{P}}: ℝ^{d}\to \\{-1,1\\}$. Thus, for any $x\in ℝ^{d}$, $\left(x, l_{\tilde{P}}(x) \right)\in \mathcal{P}$. Now, let $V:\Omega\to ℝ^{d}$ be a random vector, and thus we can define a random variable $L:= l_{\tilde{P}}(V): \left(\Omega,\mathcal{F}, ℙ \right)\to \\{-1,1 \\}$. This $L$ is called the label of $V$. If $L = 1$, we say $V$ belongs to the class to be recognized. If $L = -1$, $V$ is from any other class.  

Now, let $\hat{V} = \\{v_1,...,v_m\\}$ be the tranining samples drawn from $P$. That is, each $v_i \in \hat{V}$ is a realization of independent copy of $V$ and $\left(v_i,1\right)\in\tilde{P}$. Moreover, let $\hat{K} = \\{k_1,...,k_n \\}$ from other known classes $K\subset \tilde{P}$. That is, each $k_i \in \hat{K}$ is also a relization of independent copy of $V$ such that $(k_i,-1)\in \tilde{P}$. Furthermore, let $U$ be a subset of unknown classes, that is, $P\cup K\cup U$ is a disjoint union and $P\cup K\cup U \subset \tilde{P}$. Finally, let test data $\\{t_1,...,t_z \\}\in P\cup K\cup U$. Then, we first observe that the test data contains unknown classes and is disjoint from the training set. 

Next, let $f: ℝ^{d}\to ℝ$ be a Borel measurable recognition function. Now, we define R(f):= $ℙ\[sign(f(V)) ≠ L\]$. Equivelently, R(f) = $ℙ\[f(V)L < 0 \]$


# Part 2: Data
Training data set: https://www.bioid.com/face-database/
Validation data set: 

## Description of the Dataset
The BioID Face Database contains 1,521 grayscale images with 23 distinct human subjects
Each subject appears in multiple images under varying conditions

Instead of splitting the original dataset, I used a completely separate dataset for validation. This provides a stronger evaluation of generalization, since the model is tested on data drawn from a different distribution. The observed drop in performance indicates that the discriminator has partially overfitted to dataset-specific features rather than learning a fully distribution-invariant representation of faces.

# Part 3: Challenges:
## Lack of evaluation metric: 
One of the most immediate challenges encountered was the absence of a well-defined evaluation metric for the generative model. Unlike supervised learning tasks, where accuracy or loss directly measures performance, GANs do not provide a straightforward notion of correctness

In the current implementation, the discriminator outputs values:

D(x)∈[0,1]

which can be interpreted as the probability that an image is real. During training, I tracked:

D(x): discriminator score on real images
D(G(z)): discriminator score on generated images

However, these values are not reliable indicators of model quality. For example:

A high D(x) may simply indicate overfitting to training data
A low D(G(z)) does not necessarily imply poor generation quality
The discriminator may become too strong or too weak, distorting these signals

Thus, one challenge was interpreting these values meaningfully. Without additional metrics such as Fréchet Inception Distance (FID) or Inception Score, evaluation relied heavily on visual inspection, which is subjective and difficult to quantify.
