# Regression Methods: Principles and Memory Analysis

## Notation

| Symbol | Meaning |
|--------|---------|
| $\mathbf{X} \in \mathbb{R}^{N \times p}$ | Feature matrix, $N$ samples, $p$ features |
| $\mathbf{y} \in \mathbb{R}^{N}$ | Target vector |
| $\boldsymbol{\theta} \in \mathbb{R}^{p}$ | Coefficient vector |
| $\alpha > 0$ | Regularisation strength |
| $\lambda_1, \lambda_2$ | L1 and L2 penalty weights |

When an intercept is fitted, $\mathbf{X}$ is augmented with a column of ones, making it $N \times (p+1)$ and $\boldsymbol{\theta} \in \mathbb{R}^{p+1}$. For clarity the intercept term is absorbed into $\boldsymbol{\theta}$ throughout.

---

## 1. Ordinary Least Squares (OLS)

### Objective

Minimise the sum of squared residuals:

$$\mathcal{L}(\boldsymbol{\theta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\theta})$$

### Closed-form solution

Setting $\nabla_{\boldsymbol{\theta}} \mathcal{L} = 0$:

$$\nabla_{\boldsymbol{\theta}} \mathcal{L} = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\theta}) = \mathbf{0}$$

$$\mathbf{X}^\top \mathbf{X}\, \boldsymbol{\theta} = \mathbf{X}^\top \mathbf{y}$$

$$\boxed{\boldsymbol{\theta}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}}$$

This is the **Normal Equation**. The solution is unique when $\mathbf{X}^\top\mathbf{X}$ is invertible (i.e. $\mathbf{X}$ has full column rank).

### Why OLS is memory-bounded

Define two accumulators:

$$\mathbf{A} = \mathbf{X}^\top \mathbf{X} \in \mathbb{R}^{p \times p}, \qquad \mathbf{b} = \mathbf{X}^\top \mathbf{y} \in \mathbb{R}^{p}$$

Both are **additive** over any partition of the data. Split the dataset into $K$ batches $\{(\mathbf{X}_k, \mathbf{y}_k)\}_{k=1}^K$:

$$\mathbf{A} = \sum_{k=1}^{K} \mathbf{X}_k^\top \mathbf{X}_k, \qquad \mathbf{b} = \sum_{k=1}^{K} \mathbf{X}_k^\top \mathbf{y}_k$$

After processing batch $k$, $(\mathbf{X}_k, \mathbf{y}_k)$ can be discarded. At any point only $\mathbf{A}$ and $\mathbf{b}$ need to be held in memory. The final solve $\boldsymbol{\theta}^* = \mathbf{A}^{-1}\mathbf{b}$ requires no sample data at all.

**Memory cost: $O(p^2)$, independent of $N$.**

---

## 2. Ridge Regression (L2 Regularisation)

### Objective

Add an L2 penalty on the coefficients to shrink them toward zero:

$$\mathcal{L}_\text{ridge}(\boldsymbol{\theta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \alpha \|\boldsymbol{\theta}\|_2^2$$

The penalty discourages large coefficients and improves conditioning when $\mathbf{X}^\top\mathbf{X}$ is near-singular.

### Closed-form solution

$$\nabla_{\boldsymbol{\theta}} \mathcal{L}_\text{ridge} = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\theta}) + 2\alpha\boldsymbol{\theta} = \mathbf{0}$$

$$(\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I})\,\boldsymbol{\theta} = \mathbf{X}^\top \mathbf{y}$$

$$\boxed{\boldsymbol{\theta}^*_\text{ridge} = (\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}}$$

The matrix $\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I}$ is always positive definite for $\alpha > 0$, so the solution is always unique and numerically stable.

**Note on the intercept:** The intercept term should not be penalised. In practice, $\alpha\mathbf{I}$ is applied only to the $p \times p$ feature block of $\mathbf{X}^\top\mathbf{X}$, not to the row/column corresponding to the bias.

### Why Ridge is also memory-bounded

The only difference from OLS is adding $\alpha\mathbf{I}$ to $\mathbf{A}$ before solving. The accumulation step is identical:

$$\mathbf{A} = \sum_{k=1}^{K} \mathbf{X}_k^\top \mathbf{X}_k, \qquad \mathbf{b} = \sum_{k=1}^{K} \mathbf{X}_k^\top \mathbf{y}_k$$

After all batches are processed:

$$\boldsymbol{\theta}^*_\text{ridge} = (\mathbf{A} + \alpha \mathbf{I})^{-1} \mathbf{b}$$

No sample is ever revisited. **Memory cost: $O(p^2)$, independent of $N$.**

---

## 3. Lasso (L1 Regularisation)

### Objective

Replace the L2 penalty with an L1 penalty:

$$\mathcal{L}_\text{lasso}(\boldsymbol{\theta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \alpha \|\boldsymbol{\theta}\|_1$$

The L1 term $\|\boldsymbol{\theta}\|_1 = \sum_{j=1}^p |\theta_j|$ is not differentiable at $\theta_j = 0$, but has a subdifferential. The key property is that Lasso drives many coefficients to **exactly zero**, performing automatic feature selection.

### No closed-form solution

The subdifferential condition for optimality is:

$$-2\mathbf{x}_j^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\theta}) + \alpha \cdot \partial|\theta_j| \ni 0, \quad j = 1, \ldots, p$$

where $\partial|\theta_j|$ is the subdifferential of the absolute value:

$$\partial|\theta_j| = \begin{cases} \{+1\} & \text{if } \theta_j > 0 \\ \{-1\} & \text{if } \theta_j < 0 \\ [-1, +1] & \text{if } \theta_j = 0 \end{cases}$$

This system has no closed-form solution and is solved by **coordinate descent**: cycle over each $\theta_j$, holding all others fixed, and apply the soft-threshold update:

$$\theta_j \leftarrow \mathcal{S}_{\alpha/2}\!\left(\frac{\mathbf{x}_j^\top \mathbf{r}_{-j}}{{\|\mathbf{x}_j\|_2^2}}\right), \qquad \mathcal{S}_\tau(z) = \text{sign}(z)\max(|z| - \tau, 0)$$

where $\mathbf{r}_{-j} = \mathbf{y} - \sum_{k \neq j} \theta_k \mathbf{x}_k$ is the partial residual.

### Why Lasso cannot be batched exactly

Each coordinate update requires $\mathbf{x}_j^\top \mathbf{r}_{-j}$, the inner product of the $j$-th feature column with the current residual. The residual $\mathbf{r}_{-j}$ depends on the current $\boldsymbol{\theta}$, which changes after every coordinate update. The algorithm must **repeatedly access the full dataset** across many iterations until convergence.

More formally: suppose we split data into batches $\{(\mathbf{X}_k, \mathbf{y}_k)\}$. After processing batch $k$, we cannot discard it, because the next iteration of coordinate descent will need $\mathbf{X}_k$ again to recompute the residual under the updated $\boldsymbol{\theta}$.

There is no accumulator of fixed size $O(p^2)$ that captures all the information needed to reproduce the Lasso solution without the raw data. **Memory cost: $O(N \cdot p)$.**

---

## 4. Elastic Net (L1 + L2 Regularisation)

### Objective

Combine L1 and L2 penalties with mixing ratio $\rho \in [0, 1]$:

$$\mathcal{L}_\text{EN}(\boldsymbol{\theta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \alpha \left[ \rho \|\boldsymbol{\theta}\|_1 + \frac{1-\rho}{2} \|\boldsymbol{\theta}\|_2^2 \right]$$

- $\rho = 1$: reduces to Lasso
- $\rho = 0$: reduces to Ridge (up to a constant factor in $\alpha$)
- $0 < \rho < 1$: combines feature selection (L1) with coefficient stability (L2), useful when correlated features are present

### No closed-form solution

The L1 component again prevents a closed-form solution. ElasticNet is also solved by coordinate descent, with the update:

$$\theta_j \leftarrow \frac{1}{1 + \alpha(1-\rho)} \cdot \mathcal{S}_{\alpha\rho/2}\!\left(\frac{\mathbf{x}_j^\top \mathbf{r}_{-j}}{\|\mathbf{x}_j\|_2^2}\right)$$

The L2 term modifies the denominator, but the coordinate descent structure — and its requirement to repeatedly access all samples — is unchanged.

**Memory cost: $O(N \cdot p)$, for the same reason as Lasso.**

---

## 5. Huber Regression

### Objective

Replace the squared loss with the Huber loss, which is quadratic for small residuals and linear for large ones, making it robust to outliers:

$$\mathcal{L}_\text{Huber}(\boldsymbol{\theta}) = \sum_{i=1}^{N} \ell_\epsilon(y_i - \mathbf{x}_i^\top \boldsymbol{\theta})$$

where the Huber loss function is:

$$\ell_\epsilon(r) = \begin{cases} \frac{1}{2} r^2 & \text{if } |r| \leq \epsilon \\ \epsilon\left(|r| - \frac{\epsilon}{2}\right) & \text{if } |r| > \epsilon \end{cases}$$

The threshold $\epsilon > 0$ controls the boundary between the quadratic and linear regions. Samples with $|r_i| > \epsilon$ are treated as outliers and receive a smaller gradient signal.

### Iteratively Reweighted Least Squares (IRLS)

Huber regression is solved via IRLS. Define per-sample weights at iteration $t$:

$$w_i^{(t)} = \begin{cases} 1 & \text{if } |r_i^{(t)}| \leq \epsilon \\ \dfrac{\epsilon}{|r_i^{(t)}|} & \text{if } |r_i^{(t)}| > \epsilon \end{cases}$$

where $r_i^{(t)} = y_i - \mathbf{x}_i^\top \boldsymbol{\theta}^{(t)}$ is the residual at the current iterate. Let $\mathbf{W}^{(t)} = \text{diag}(w_1^{(t)}, \ldots, w_N^{(t)})$. Each iteration solves a weighted least squares problem:

$$\boldsymbol{\theta}^{(t+1)} = \left(\mathbf{X}^\top \mathbf{W}^{(t)} \mathbf{X}\right)^{-1} \mathbf{X}^\top \mathbf{W}^{(t)} \mathbf{y}$$

### Why Huber cannot be batched exactly

The weights $w_i^{(t)}$ depend on the current residuals $r_i^{(t)} = y_i - \mathbf{x}_i^\top\boldsymbol{\theta}^{(t)}$, and $\boldsymbol{\theta}^{(t)}$ changes at every iteration. Therefore:

1. At iteration $t$, computing $\mathbf{X}^\top \mathbf{W}^{(t)} \mathbf{X}$ requires visiting every sample to evaluate $w_i^{(t)}$.
2. At iteration $t+1$, the weights change, so the same samples must be visited again under the new $\boldsymbol{\theta}^{(t+1)}$.

Unlike OLS and Ridge, the matrix $\mathbf{X}^\top \mathbf{W}^{(t)} \mathbf{X}$ cannot be pre-accumulated once and reused across iterations, because $\mathbf{W}^{(t)}$ is different at each step. **Memory cost: $O(N \cdot p)$.**

---

## Summary

| Method | Has closed-form solution | Accumulator structure | Memory cost |
|--------|--------------------------|----------------------|-------------|
| OLS | Yes | $\mathbf{X}^\top\mathbf{X}$, $\mathbf{X}^\top\mathbf{y}$ fixed after one pass | $O(p^2)$ |
| Ridge | Yes | Same as OLS, $+\,\alpha\mathbf{I}$ at solve time | $O(p^2)$ |
| Lasso | No | Coordinate descent; residuals depend on current $\boldsymbol{\theta}$ | $O(N \cdot p)$ |
| ElasticNet | No | Same as Lasso | $O(N \cdot p)$ |
| Huber | No | IRLS; weights depend on current $\boldsymbol{\theta}$ | $O(N \cdot p)$ |

The fundamental distinction is whether the optimality condition can be expressed as a **linear system in fixed sufficient statistics** ($\mathbf{X}^\top\mathbf{X}$ and $\mathbf{X}^\top\mathbf{y}$) that are additive over batches. OLS and Ridge satisfy this; Lasso, ElasticNet, and Huber do not.
