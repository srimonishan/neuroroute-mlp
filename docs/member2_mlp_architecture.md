## 2. MLP Model Implementation & Mathematical Formulation (Member 2 - Zaid)

### 2.1 Network Architecture

The NeuroRoute decision block is implemented in NumPy (`src/models/mlp.py`) for fixed-size $n = 20$ TSP next-city classification:

* **Input Layer ($d_0 = 3n = 60$):** Takes the state vector $x_t \in \mathbb{R}^{60}$ built by Member 1's pipeline (20-dim current-city one-hot vector, 20-dim visited mask, 20-dim instance-normalized distances).
* **Hidden Layer 1 ($d_1 = 128$):** Dense layer with **ReLU** activation (`He` normal initialization: $W^{(1)}_{ij} \sim \mathcal{N}(0, 2/d_0)$, $b^{(1)} = 0$).
* **Hidden Layer 2 ($d_2 = 64$):** Dense layer with **ReLU** activation (`He` normal initialization: $W^{(2)}_{ij} \sim \mathcal{N}(0, 2/d_1)$, $b^{(2)} = 0$).
* **Output Layer ($d_3 = n = 20$):** Dense layer (`Xavier/Glorot` uniform initialization: $W^{(3)}_{ij} \sim \mathcal{U}[-\sqrt{6/(d_2+d_3)}, \sqrt{6/(d_2+d_3)}]$, $b^{(3)} = 0$) outputting logits $z^{(3)} \in \mathbb{R}^{20}$.

Total trainable parameters for `60 -> 128 -> 64 -> 20`:
* **Weights ($W$):** $60 \times 128 + 128 \times 64 + 64 \times 20 = 7,680 + 8,192 + 1,280 = 17,152$
* **Biases:** $128 + 64 + 20 = 212$
* **Total parameters ($P$):** $17,364$

`src/models/activations.py` also includes `LeakyReLU`, `Sigmoid`, `Tanh`, and `Softmax` with matching `forward` and `backward` methods so different architectures can be tested without changing the training loop.

---

### 2.2 Forward Propagation & Numerically Stable Loss

For a mini-batch $X = A^{(0)} \in \mathbb{R}^{B \times d_0}$ across $L$ dense layers ($l = 1, \dots, L$):

$$Z^{(l)} = A^{(l-1)} W^{(l)} + \mathbf{1}_B (b^{(l)})^\top$$

$$A^{(l)} = g^{(l)}(Z^{(l)}) \quad \text{for } l = 1, \dots, L - 1$$

where $g(z) = \max(0, z)$ for ReLU, and the output layer produces logits $Z^{(L)} \in \mathbb{R}^{B \times C}$ ($C = 20$).

To avoid overflow when exponentiating large logits, `SoftmaxCrossEntropyLoss` (`src/models/losses.py`) subtracts the row maximum $m_i = \max_c Z^{(L)}_{i,c}$ before applying log-softmax:

$$\log \hat{p}_{i,c} = (Z^{(L)}_{i,c} - m_i) - \log \sum_{j=0}^{C-1} \exp(Z^{(L)}_{i,j} - m_i)$$

$$\hat{p}_{i,c} = \exp(\log \hat{p}_{i,c})$$

Given target class $y_i \in \{0, \dots, C-1\}$, the batch-mean cross-entropy loss is:

$$\mathcal{L}(Z^{(L)}, y) = -\frac{1}{B} \sum_{i=1}^{B} \log \hat{p}_{i, y_i}$$

---

### 2.3 Backpropagation Equations

Combining softmax and cross-entropy gives the output-layer gradient:

$$\delta^{(L)}_{i,c} = \frac{\partial \mathcal{L}}{\partial Z^{(L)}_{i,c}} = \frac{1}{B}\left(\hat{p}_{i,c} - \mathbb{I}[c = y_i]\right)$$

Propagating backwards through layers $l = L, L-1, \dots, 1$:

$$\nabla_{W^{(l)}} \mathcal{L} = (A^{(l-1)})^\top \delta^{(l)} \in \mathbb{R}^{d_{l-1} \times d_l}$$

$$\nabla_{b^{(l)}} \mathcal{L} = \sum_{i=1}^{B} \delta^{(l)}_i \in \mathbb{R}^{d_l}$$

$$\delta^{(l-1)} = \left(\delta^{(l)} (W^{(l)})^\top\right) \odot (g^{(l-1)})'(Z^{(l-1)}) \quad (\text{for } l > 1)$$

Since $\delta^{(L)}$ already includes the $\frac{1}{B}$ batch average, parameter gradients are directly batch-averaged. All analytic gradients (`dW`, `db`, `dX`, activations, and loss) are checked against central finite differences in `tests/test_mlp.py`.

---

### 2.4 Optimizer Interface

To allow controlled comparison across SGD, Momentum, Adam, RMSprop, and AdaGrad (Member 3), `MLP` provides:

* `named_parameters()` / `parameters()`: returns mutable `(param, grad)` NumPy arrays for in-place updates.
* `zero_grad()`: clears `dW` and `db` before each step.
* `get_weights()` / `set_weights(weights)`: saves and restores exact weight copies so all five optimizers start from identical initial weights.
* `save_weights(path)` / `load_weights(path)`: `.npz` checkpoint saving and loading.
* `train()` / `eval()`: switches between training and evaluation modes.

---

### 2.5 Route Decoding & Constraint Validation

Because per-step classification accuracy alone does not enforce TSP constraints, `src/routing/decoder.py` decodes closed tours step by step (`decode_tour`):

1. Start at city `0` with `visited = [1, 0, ..., 0]`.
2. At each step $t = 0, \dots, n-2$, build $x_t \in \mathbb{R}^{3n}$ and compute logits $z_t = \text{MLP}(x_t)$.
3. Mask out already-visited cities (and any non-finite or negative edges) by setting their logits to $-\infty$, then pick $v_{t+1} = \arg\max_c z_{t,c}^{\text{masked}}$.
4. Return to `start`, validate Hamiltonian constraints with `validate_tour()`, and compute closed tour cost with `tour_length()`.

---

### 2.6 Time & Space Complexity (Forward, Backward & Decoding)

Let $B$ be batch size, $N$ training samples, $E$ epochs, $L$ dense layers with widths $d_0, d_1, \dots, d_L$, $W = \sum_{l=1}^L d_{l-1} d_l$ dense weights, and $P = \sum_{l=1}^L (d_{l-1} d_l + d_l)$ total trainable parameters.

| Operation | Time Complexity | Space Complexity | Notes |
| --- | --- | --- | --- |
| **Forward pass (1 batch)** | $O\!\left(B \sum_{l=1}^L d_{l-1} d_l\right) = O(BW)$ | $O\!\left(B \sum_{l=0}^L d_l + P\right)$ | Matrix multiplications $A^{(l-1)}W^{(l)}$ dominate over $O(B \sum_{l=1}^L d_l)$ bias, activation, and softmax ops; caches inputs and pre-activations for backprop. |
| **Backward pass (1 batch)** | $O\!\left(B \sum_{l=1}^L d_{l-1} d_l\right) = O(BW)$ | $O\!\left(B \sum_{l=0}^L d_l + 2P\right)$ | Computes $(A^{(l-1)})^\top \delta^{(l)}$ and $\delta^{(l)} (W^{(l)})^\top$ per layer while storing weights and gradients ($2P$). |
| **Full training ($E$ epochs)** | $O\!\left(E \cdot N \cdot W + E \left\lceil\frac{N}{B}\right\rceil P\right)$ | $O\!\left(B \sum_{l=0}^L d_l + (2 + k_{\text{opt}}) P\right)$ | $k_{\text{opt}}$ is the number of optimizer state buffers per parameter (0 for SGD, 1 for Momentum/RMSprop/AdaGrad, 2 for Adam). |
| **Tour decoding (1 instance, $n$ cities)** | $O(n \cdot W + n^2)$ | $O(\sum_{l=0}^L d_l + P + n^2)$ | Performs $n-1$ single-sample ($B=1$) forward passes ($O(nW)$) plus $O(n)$ feature building and masking per step ($O(n^2)$ total). |
