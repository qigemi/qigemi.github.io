---
layout: post
---

这部分比较杂，我就把剩下几节课放到一起了。

#17.对称矩阵
我们之前说过对称矩阵的特征值都是实数，我少说一句，应该是实对称矩阵。**对于实对称矩阵，它的特征值都是实数，而且它的特征向量互相正交**。这里证明一下。首先对任意矩阵

$$Ax=\lambda x\Rightarrow\bar A\bar x=\bar\lambda\bar x$$

上横线表示复数共轭，

$$\begin{align}\bar A\bar x=\bar\lambda\bar x\Rightarrow \bar x^T\bar A^T=\bar x^T\bar\lambda^T \Rightarrow \bar x^T\bar A^Tx=&\bar\lambda^T\bar x^Tx \\
Ax=\lambda x\Rightarrow \bar x^TAx=&\lambda^T\bar x^Tx\end{align}$$

比较上下两式，$\bar x^Tx$ 其实就是向量x的长度平方，它是大于0的。分如下几种情况：

1. $\bar A^T=A\Rightarrow \lambda=\bar\lambda$，实对称矩阵和复共轭对称矩阵特征值是实数
2. $\bar A^T=-A\Rightarrow \lambda=-\bar\lambda$，反对称矩阵特征值是纯虚数

对于实对称的矩阵，其特征值都是实数，证毕。特征向量互相正交课里没讲，只好自己证明一下。

$$Ax_1=\lambda_1x_1 \\
Ax_2=\lambda_2x_2$$

一式取转置右乘x2

$$\Rightarrow x_1^TA=\lambda_1x_1^T \\
\Rightarrow x_1^TAx_2=\lambda_1x_1^Tx_2 \\
\Rightarrow x_1^T\lambda_2x_2=\lambda_1x_1^Tx_2 \\
\Rightarrow (\lambda_1-\lambda_2)x_1^Tx_2=0$$

如果是互异的两个特征值，则两个向量正交，如果是同一个特征值对应的不同特征向量，我们可以在零空间中选择正交的一组基作为特征向量。这样所有的特征向量就互相垂直了。woohoo。如果A可以对角化则还有

$$A=Q\Lambda Q^{-1}=Q\Lambda Q^T$$

不是卖萌，Q是单位正交矩阵。这是从对角化公式推广来的。
