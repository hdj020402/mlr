# 回归方法原理与内存分析

## 符号说明

| 符号 | 含义 |
|------|------|
| $\mathbf{X} \in \mathbb{R}^{N \times p}$ | 特征矩阵，$N$ 个样本，$p$ 个特征 |
| $\mathbf{y} \in \mathbb{R}^{N}$ | 目标向量 |
| $\boldsymbol{\theta} \in \mathbb{R}^{p}$ | 系数向量 |
| $\alpha > 0$ | 正则化强度 |
| $\rho \in [0,1]$ | ElasticNet 中 L1 与 L2 的混合比例 |

当需要拟合截距时，$\mathbf{X}$ 被扩充一列全 1，变为 $N \times (p+1)$，截距被吸收进 $\boldsymbol{\theta}$。下文为书写简洁，统一采用此约定。

---

## 1. 普通最小二乘（OLS）

### 目标函数

最小化残差平方和：

$$\mathcal{L}(\boldsymbol{\theta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\theta})$$

### 闭合解推导

对 $\boldsymbol{\theta}$ 求梯度并令其为零：

$$\nabla_{\boldsymbol{\theta}} \mathcal{L} = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\theta}) = \mathbf{0}$$

整理得**正规方程（Normal Equations）**：

$$\mathbf{X}^\top \mathbf{X}\, \boldsymbol{\theta} = \mathbf{X}^\top \mathbf{y}$$

$$\boxed{\boldsymbol{\theta}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}}$$

当 $\mathbf{X}$ 列满秩时，$\mathbf{X}^\top\mathbf{X}$ 可逆，解唯一。

### 为什么 OLS 可以省内存

定义两个累加器：

$$\mathbf{A} = \mathbf{X}^\top \mathbf{X} \in \mathbb{R}^{p \times p}, \qquad \mathbf{b} = \mathbf{X}^\top \mathbf{y} \in \mathbb{R}^{p}$$

关键性质：**$\mathbf{A}$ 和 $\mathbf{b}$ 对数据的任意划分都是可加的**。将数据分为 $K$ 个批次 $\{(\mathbf{X}_k, \mathbf{y}_k)\}_{k=1}^K$：

$$\mathbf{A} = \sum_{k=1}^{K} \mathbf{X}_k^\top \mathbf{X}_k, \qquad \mathbf{b} = \sum_{k=1}^{K} \mathbf{X}_k^\top \mathbf{y}_k$$

处理完第 $k$ 批后，$(\mathbf{X}_k, \mathbf{y}_k)$ 即可丢弃，内存中只需保留 $\mathbf{A}$ 和 $\mathbf{b}$ 这两个小矩阵。最终求解 $\boldsymbol{\theta}^* = \mathbf{A}^{-1}\mathbf{b}$ 完全不需要原始数据。

**内存开销：$O(p^2)$，与样本量 $N$ 无关。**

---

## 2. 岭回归（Ridge，L2 正则化）

### 目标函数

在 OLS 基础上加入 L2 惩罚项，对系数的大小加以约束：

$$\mathcal{L}_\text{ridge}(\boldsymbol{\theta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \alpha \|\boldsymbol{\theta}\|_2^2$$

L2 惩罚使所有系数向零收缩，同时改善 $\mathbf{X}^\top\mathbf{X}$ 近奇异时的数值条件数。

### 闭合解推导

$$\nabla_{\boldsymbol{\theta}} \mathcal{L}_\text{ridge} = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\theta}) + 2\alpha\boldsymbol{\theta} = \mathbf{0}$$

整理得：

$$(\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I})\,\boldsymbol{\theta} = \mathbf{X}^\top \mathbf{y}$$

$$\boxed{\boldsymbol{\theta}^*_\text{ridge} = (\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}}$$

对任意 $\alpha > 0$，矩阵 $\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I}$ 是正定矩阵，解总是唯一且数值稳定。

**关于截距的处理：** 截距项不应被正则化，因此 $\alpha\mathbf{I}$ 只作用于 $\mathbf{X}^\top\mathbf{X}$ 对应特征的 $p \times p$ 子块，截距所在的行列不加惩罚。

### 为什么 Ridge 同样可以省内存

与 OLS 完全相同的累加过程：

$$\mathbf{A} = \sum_{k=1}^{K} \mathbf{X}_k^\top \mathbf{X}_k, \qquad \mathbf{b} = \sum_{k=1}^{K} \mathbf{X}_k^\top \mathbf{y}_k$$

累加完成后，仅在求解前对对角线施加惩罚：

$$\boldsymbol{\theta}^*_\text{ridge} = (\mathbf{A} + \alpha \mathbf{I})^{-1} \mathbf{b}$$

$\alpha\mathbf{I}$ 是一个常数矩阵，与数据无关，无需任何样本参与这一步。**每个批次处理完即可丢弃，内存开销：$O(p^2)$，与 $N$ 无关。**

---

## 3. Lasso（L1 正则化）

### 目标函数

将 L2 惩罚换为 L1 惩罚：

$$\mathcal{L}_\text{lasso}(\boldsymbol{\theta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \alpha \|\boldsymbol{\theta}\|_1$$

其中 $\|\boldsymbol{\theta}\|_1 = \sum_{j=1}^p |\theta_j|$。L1 惩罚的几何性质使得许多系数被精确压缩至零，从而实现**自动特征选择**。

### 无闭合解

$|\theta_j|$ 在 $\theta_j = 0$ 处不可微，其次微分为：

$$\partial|\theta_j| = \begin{cases} \{+1\} & \theta_j > 0 \\ \{-1\} & \theta_j < 0 \\ [-1, +1] & \theta_j = 0 \end{cases}$$

最优性条件（KKT 条件）为：

$$-2\mathbf{x}_j^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\theta}) + \alpha \cdot s_j = 0, \quad s_j \in \partial|\theta_j|$$

该方程组没有解析解，通常用**坐标下降法**求解：固定其他系数，对第 $j$ 个系数应用软阈值更新：

$$\theta_j \leftarrow \mathcal{S}_{\alpha/2}\!\left(\frac{\mathbf{x}_j^\top \mathbf{r}_{-j}}{\|\mathbf{x}_j\|_2^2}\right), \qquad \mathcal{S}_\tau(z) = \text{sign}(z)\max(|z|-\tau,\, 0)$$

其中 $\mathbf{r}_{-j} = \mathbf{y} - \sum_{k \neq j} \theta_k \mathbf{x}_k$ 是去掉第 $j$ 个特征后的偏残差。

### 为什么 Lasso 无法分批

每次坐标更新需要计算 $\mathbf{x}_j^\top \mathbf{r}_{-j}$，而偏残差 $\mathbf{r}_{-j}$ 依赖**当前的** $\boldsymbol{\theta}$。坐标下降每更新一个 $\theta_j$，残差就发生变化，下一步更新 $\theta_{j+1}$ 又需要在新的 $\boldsymbol{\theta}$ 下重新计算残差。

更严格地说：假设将数据分批，处理完第 $k$ 批后将其丢弃，则第 $k+1$ 批处理完毕、$\boldsymbol{\theta}$ 更新后，**已经丢弃的第 $k$ 批数据无法再参与下一轮迭代的残差计算**，导致结果错误。

不存在一个大小为 $O(p^2)$ 的固定累加器，能在不保留原始数据的情况下复现 Lasso 的精确解。**内存开销：$O(N \cdot p)$。**

---

## 4. Elastic Net（L1 + L2 混合正则化）

### 目标函数

以混合比例 $\rho \in [0,1]$ 结合 L1 与 L2 惩罚：

$$\mathcal{L}_\text{EN}(\boldsymbol{\theta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \alpha \left[ \rho \|\boldsymbol{\theta}\|_1 + \frac{1-\rho}{2} \|\boldsymbol{\theta}\|_2^2 \right]$$

- $\rho = 1$：退化为 Lasso
- $\rho = 0$：退化为 Ridge（差一个常数因子）
- $0 < \rho < 1$：同时具备稀疏性（L1）和系数稳定性（L2），适合特征间存在相关性的场景

### 无闭合解

L1 项的存在同样使方程组无解析解。坐标下降更新为：

$$\theta_j \leftarrow \frac{1}{1 + \alpha(1-\rho)} \cdot \mathcal{S}_{\alpha\rho/2}\!\left(\frac{\mathbf{x}_j^\top \mathbf{r}_{-j}}{\|\mathbf{x}_j\|_2^2}\right)$$

L2 项改变了分母，但坐标下降的整体结构——以及它对全量数据的反复访问需求——与 Lasso 完全相同。

**内存开销：$O(N \cdot p)$，原因与 Lasso 相同。**

---

## 5. Huber 回归

### 目标函数

用 Huber 损失替代平方损失。Huber 损失对小残差使用二次项（与 OLS 相同），对大残差使用线性项（减小异常值的影响）：

$$\mathcal{L}_\text{Huber}(\boldsymbol{\theta}) = \sum_{i=1}^{N} \ell_\epsilon(y_i - \mathbf{x}_i^\top \boldsymbol{\theta})$$

$$\ell_\epsilon(r) = \begin{cases} \dfrac{1}{2} r^2 & |r| \leq \epsilon \\ \epsilon\!\left(|r| - \dfrac{\epsilon}{2}\right) & |r| > \epsilon \end{cases}$$

阈值 $\epsilon > 0$ 决定"异常值"的判定边界。$|r_i| > \epsilon$ 的样本被视为异常值，其梯度信号被线性截断。

### 迭代重加权最小二乘（IRLS）

Huber 回归通过 IRLS 求解。在第 $t$ 次迭代时，为每个样本定义权重：

$$w_i^{(t)} = \begin{cases} 1 & |r_i^{(t)}| \leq \epsilon \\ \dfrac{\epsilon}{|r_i^{(t)}|} & |r_i^{(t)}| > \epsilon \end{cases}$$

其中 $r_i^{(t)} = y_i - \mathbf{x}_i^\top \boldsymbol{\theta}^{(t)}$ 是当前迭代的残差。令 $\mathbf{W}^{(t)} = \text{diag}(w_1^{(t)}, \ldots, w_N^{(t)})$，每轮迭代求解一个加权最小二乘问题：

$$\boldsymbol{\theta}^{(t+1)} = \left(\mathbf{X}^\top \mathbf{W}^{(t)} \mathbf{X}\right)^{-1} \mathbf{X}^\top \mathbf{W}^{(t)} \mathbf{y}$$

重复直至 $\|\boldsymbol{\theta}^{(t+1)} - \boldsymbol{\theta}^{(t)}\|$ 满足收敛准则。

### 为什么 Huber 无法分批

注意 $\mathbf{X}^\top \mathbf{W}^{(t)} \mathbf{X}$ 可以写成：

$$\mathbf{X}^\top \mathbf{W}^{(t)} \mathbf{X} = \sum_{i=1}^{N} w_i^{(t)} \mathbf{x}_i \mathbf{x}_i^\top$$

表面上与 OLS 类似（$w_i \equiv 1$ 时退化为 $\mathbf{X}^\top\mathbf{X}$），但关键区别在于：**权重 $w_i^{(t)}$ 依赖当前残差，而残差依赖当前参数 $\boldsymbol{\theta}^{(t)}$**，并且 $\boldsymbol{\theta}^{(t)}$ 在每轮迭代后都会改变。

因此：
1. 第 $t$ 轮需要访问全量数据，计算每个样本在当前 $\boldsymbol{\theta}^{(t)}$ 下的权重；
2. 第 $t+1$ 轮 $\boldsymbol{\theta}$ 更新后，权重随之改变，必须重新访问全量数据。

与 OLS 不同，$\mathbf{X}^\top \mathbf{W}^{(t)} \mathbf{X}$ 无法在迭代间复用，每次迭代都是一次对全量数据的完整遍历。**内存开销：$O(N \cdot p)$。**

---

## 总结

| 方法 | 是否有闭合解 | 充分统计量结构 | 内存开销 |
|------|------------|--------------|---------|
| OLS | 是 | $\mathbf{X}^\top\mathbf{X}$，$\mathbf{X}^\top\mathbf{y}$，一次遍历后固定 | $O(p^2)$ |
| Ridge | 是 | 同 OLS，求解前加 $\alpha\mathbf{I}$ | $O(p^2)$ |
| Lasso | 否 | 坐标下降，残差依赖当前 $\boldsymbol{\theta}$ | $O(N \cdot p)$ |
| ElasticNet | 否 | 同 Lasso | $O(N \cdot p)$ |
| Huber | 否 | IRLS，权重依赖当前 $\boldsymbol{\theta}$ | $O(N \cdot p)$ |

**根本区别** 在于最优性条件能否被表示为关于**固定充分统计量**（$\mathbf{X}^\top\mathbf{X}$ 和 $\mathbf{X}^\top\mathbf{y}$）的线性方程组，而这些统计量对批次具有可加性。OLS 和 Ridge 满足这一条件；Lasso、ElasticNet 和 Huber 由于引入了与当前参数耦合的非线性结构，不满足这一条件。
