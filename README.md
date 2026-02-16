# Neural-Network-Project

## The Problem 
The goal of this exposition is to state the Open Set Recognition problem rigorously and to discuss possible solutions to tackle it. 

Let $\left(\Omega,\mathcal{F}, ℙ \right)$ be the underlying probability space and let partition $\tilde{\mathcal{P}}$ over $ℝ^{d}\times \\{-1,1\\}$ be a class. Then such a partition $\tilde{\mathcal{P}}$ induces a function $l_{\tilde{P}}: ℝ^{d}\to \\{-1,1\\}$. Thus, for any $x\in ℝ^{d}$, $\left(x, l_{\tilde{P}}(x) \right)\in \mathcal{P}$. Now, let $V:\Omega\to ℝ^{d}$ be a random vector, and thus we can define a random variable $L:= l_{\tilde{P}}(V): \left(\Omega,\mathcal{F}, ℙ \right)\to \\{-1,1 \\}$. This $L$ is called the label of $V$. If $L = 1$, we say $V$ belongs to the class to be recognized. If $L = -1$, $V$ is from any other class.  

Now, let $\hat{V} = \\{v_1,...,v_m\\}$ be the tranining samples drawn from $P$. That is, each $v_i \in \hat{V}$ is a realization of independent copy of $V$ and $\left(v_i,1\right)\in\tilde{P}$. Moreover, let $\hat{K} = \\{k_1,...,k_n \\}$ from other known classes $K\subset \tilde{P}$. That is, each $k_i \in \hat{K}$ is also a relization of independent copy of $V$ such that $(k_i,-1)\in \tilde{P}$. Furthermore, let $U$ be a subset of unknown classes, that is, $P\cup K\cup U$ is a disjoint union and $P\cup K\cup U \subset \tilde{P}$. Finally, let test data $\\{t_1,...,t_z \\}\in P\cup K\cup U$. Then, we first observe that the test data contains unknown classes and is disjoint from the training set. 

Next, let $f: ℝ^{d}\to ℝ$ be a Borel measurable recognition function. Now, we define R(f):= $ℙ\[sign(f(V)) ≠ L\]$. Equivelently, R(f) = $ℙ\[f(V)L < 0 \]$
