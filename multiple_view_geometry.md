---
layout: pose
---
-------------
# 基本矩阵和本质矩阵
\[
E = [t]_{\times}R
\]
其中
\[
[t]_{\times} =  
\left[
 \begin{matrix}
   0 & -t_3 & t_2 \\
   t_3 & 0 & -t_1 \\
   -t_2 & t_1 & 0
  \end{matrix}
  \right]
\]
\[
R = R_\alpha R_\beta R_\gamma =
\left[
 \begin{matrix}
   1 & 0 & 0 \\
   0 & \cos\alpha & -\sin\alpha \\
   0 & \sin\alpha & \cos\alpha
  \end{matrix}
\right]
\left[
 \begin{matrix}
   \cos\beta & 0 & \sin\beta \\
   0 & 1 & 0 \\
   -\sin\beta & 0 & \cos\beta
  \end{matrix}
\right]
\left[
 \begin{matrix}
   \cos\gamma & -\sin\gamma & 0 \\
   \sin\gamma & \cos\gamma & 0 \\
   0 & 0 & 1
  \end{matrix}
\right]
\]
从1到2的转换角度为 $\alpha, \beta, \gamma$，则坐标转换关系为 $S_1 = RS_2$。

\[
P=K[I|0] \qquad P^{\prime}=K^{\prime}[R|t]\\
F={[e^{\prime}]}_{\times}K^{\prime}RK^{-1}=K^{\prime-T} {[t]}_{\times} R K^{-1}
\]

对左右图像中的点，满足

$$x^{\prime T}Fx=0\\
x^{\prime T}Ex=0$$

## 计算基本矩阵
### 1.基本方程
由 $x^{\prime T}Fx=0$ 及多组点对应可以列出线性方程组 $Af=0, A\in \mathbb R^{n\times 9}$ , n为匹配点数。如果A的秩为8，则可以求A的零空间得到f。如果A的秩为9，则通过最小二乘方法求

$$
\begin{align}
minimize\quad  &\|Af\|\\
subject\ to\quad  &\|f\|=1
\end{align} \tag{1}
$$

该方法称为八点法。求得的F可能不是奇异矩阵，这时通过奇异值分解，将最小的奇异值置0，得到满足条件的F。

如果匹配点数只有7个，则A的秩为7,$Af=0$ 的解是 $\alpha F_1+(1-\alpha )F_2$ 构成的二维空间，通过F的奇异性约束 $det(\alpha F_1+(1-\alpha )F_2) = 0$ 列出 $\alpha$ 的三次多项式方程，解出三个F。

### 2.归一化八点法
将匹配点图像坐标归一化到 $(0,\sqrt2)$ ，再用八点法计算，好处是解比较稳定。

### 3.代数最小化算法
上述解线性方程方法需要强制约束F为奇异矩阵（方法是奇异值分解），这不是最优的数值方法。另一种方法是直接将F表示为奇异矩阵的形式 $F=M{[e]}_{\times}$ ，其中M是非奇异矩阵，${[e]}_{\times}$ 是任意反对称矩阵。在这种表示下做最优化式 $(1)$ 。

### 4.几何距离
