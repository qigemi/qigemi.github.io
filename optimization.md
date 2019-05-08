---
layout: post
---

# 单纯形法
## 基本解
对于有不等式约束的线性规划：

$$
\begin{align}
minimize\quad  &\boldsymbol{c}^T\boldsymbol b\\
subject\ to\quad  &\boldsymbol{Ax = b}\\
 &\boldsymbol{x \geqslant 0}
\end{align}
$$
$A \in \mathbb R^{m\times n}, b \in \mathbb R^m, m<n, rank\ A = m$

解方程时可通过消元法，将线性无关的m列放在A的前面，即可写成$A = [B, D]$，通过求解 $\boldsymbol{Bx_B=b}$ 可得 $\boldsymbol{Ax = b}$ 的一个解 $\boldsymbol{x=[x_B^T,0^T]^T}$。则该解称为基本解，通过调整矩阵B，可以得到不同的基本解。如果基本解 $x\geqslant 0$，则称为基本可行解。

目标函数的最优值总是可以在某个基本可行解上得到，每个基本可行解对应一个可行域上的极点。

## 单纯形法
单纯形法的思想就是从一个基本可行解变换到另一个基本可行解，直到找到最优解。
假设一组基向量 $a_1,...,a_m$，对应的基本解 $x=[y_{10},...,y_{m0},0,...,0]^T$ 是可行的，下面要让另一个向量 $a_q (q>m)$ 变成基向量，而其中一个基向量退基，将$a_q$表示成现在基向量的线性组合，并在两边同乘 $\epsilon>0$
$$
\epsilon\boldsymbol a_q = \epsilon y_{1q}\boldsymbol a_1 + ... + \epsilon y_{mq}\boldsymbol a_m
$$
基本解带入方程 $\boldsymbol{Ax = b}$ 并和上式联立得
\[
(y_{10}-\epsilon y_{1q})\boldsymbol a_1 + ... + (y_{m0}-\epsilon y_{mq})\boldsymbol a_m+\epsilon\boldsymbol a_q = \boldsymbol b
\]
显然向量
\[
[y_{10}-\epsilon y_{1q},...,y_{m0}-\epsilon y_{mq},...,0,...,\epsilon,...,0]^T
\]
为方程的一个解，改变 $\epsilon$ 的值，直到前m个元素中第一次出现0,则得到一个新的基本可行解。
