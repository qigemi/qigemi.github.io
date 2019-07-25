---
layout: post
---

这部分比较杂，我就把剩下几节课放到一起了。

# 17.对称矩阵
我们之前说过对称矩阵的特征值都是实数，我少说一句，应该是实对称矩阵。**对于实对称矩阵，它的特征值都是实数，而且它的特征向量互相正交**。这里证明一下。注意除了下面的证明涉及到了复数矩阵，其他时候我们都只考虑实数矩阵。首先对任意矩阵

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

不是卖萌，Q是单位正交矩阵。这是从对角化公式推广来的。如果你把它展开就会发现对称矩阵是一些列投影矩阵的线性组合。对称矩阵还有一条性质，其正主元个数等于正特征值个数。这个有啥用呢？高维矩阵特征值用之前的方法求计算量很大，利用上述性质的数值计算方法会比较方便，比如我们将原矩阵减去n倍的单位阵，再求主元，就可以知道原矩阵特征值中有几个大于n，几个小于n。

## 复数
我们只在这一节介绍复数的情况，这之前和之后都是实数矩阵。之后再讨论特征值和特征向量时也都是实数的，这意味着后面我们将主要讨论实对称矩阵。

首先对于复向量 $z_i\in \mathbb C^n$，内积定义为 $\bar z_i^Tz_j$，向量长度是 $|z|=\bar z^Tz$。如上面证明中的内容，“好矩阵”不再是 $A=A^T$，而是 $A=\bar A^T=A^H$，其中共轭转置记为 $A^H$。一组单位正交基的关系也变为 $\bar q_i^Tq_j=0,\bar q_i^Tq_i=1$。满足关系 $Q^HQ=1$ 的矩阵叫做酉阵（unitary），对应于实数矩阵的单位正交阵。

复矩阵的一个例子是傅里叶变换矩阵，感兴趣的请看视频，我不想写了…

# 18.正定
正定矩阵（positive definite）首先是对称矩阵，它的特征值不仅是实数，还都是正的，当然主元也都是正的了。判断矩阵是正定的有几种方法，当然它首先得是对称矩阵：

1. 特征值都大于0
2. 主元都大于0
3. 所有子行列式都大于0（指矩阵左上角的所有方阵）
4. 二次型 $x^TAx>0，x\neq 0$

我们最常见的定义是4，实际上前三条都和第四条等价，下面简要证明一下。这里从二维矩阵例子开始，矩阵正定需要满足：

$$A=\left[\begin{matrix}a&b\\b&c\end{matrix}\right],x=\left[\begin{matrix}x\\y\end{matrix}\right]\\
\begin{align}x^TAx&=ax^2+2bxy+cy^2\\
&=a(x+\frac{b}{a}y)^2+(\frac{ac-b^2}{a})y^2>0\\
&\Rightarrow a>0,\frac{ac-b^2}{a}>0\end{align}$$

这两个结果刚好是两个子行列式，这说明方法3和4是等价的。另外看看A消元的结果：

$$A=\left[\begin{matrix}a&b\\0&c-\frac{b^2}{a}\end{matrix}\right]$$

这说明上述判断条件又刚好等于主元大于0，和方法2也是等价的。二次型配方后的结果刚好是消元系数和消元后的主元，可以说二次型配方和消元是等价的，配方后平方项系数就是主元。有没有头皮发麻。4和1的等价我瞎证一下试试：

$$x^TAx=x^TQ^T\Lambda Qx=(Qx)^T\Lambda Qx>0$$

因为对角阵中元素都是特征值，也是主元，它们都大于0，所以正定。好了，这样1234都等价了。

说了这么多，正定矩阵常见吗。最小二乘中的 $A^TA$ 就是一个正定矩阵，首先它是一个对称矩阵，然后 $x^TA^TAx=(Ax)^TAx=|Ax|$，当A为列满秩时上式在 $x\neq 0$ 的情况下恒大于零，$A^TA$ 正定。

## 特征值和椭球长短轴
我们可以再观察一下矩阵特征值和二次型函数图像之间的联系。二次函数 $f(x,y)=ax^2+2bxy+cy^2$ 是过原点的二次曲面，它可能是抛物面或者双曲面，实际上它的形状取决于矩阵A的特征值和特征向量。我本来还想写出二次曲线的表达式再分解求出椭圆的长短轴之类的，后来发现实在太难了，就要放弃的时候我想到了基变换（然后发现视频里其实提了一嘴主轴定理……），其实我上面已经写出这个公式了：

$$x^TAx=x^TQ^T\Lambda Qx=(Qx)^T\Lambda Qx\\
=\left[\begin{matrix}x{'}_1\ x{'}_2\cdots x{'}_n\end{matrix}\right]
\left[\begin{matrix}\lambda_1 & & \\ &\ddots&\\&&\lambda_n\end{matrix}\right]
\left[\begin{matrix}x{'}_1 \\ x{'}_2 \\ \vdots \\ x{'}_n\end{matrix}\right]\\
=\lambda_1x{'}_1^2+\lambda_2x{'}_2^2+\cdots +\lambda_nx{'}_n^2$$

这是什么啊朋友们，我们把坐标系转换到特征向量（它们都是相互垂直的）上，就得到标准二次曲线公式了，长短轴一目了然，高维情况也是如此，比如三维正定矩阵表示一个椭球，三个特征向量就是椭球对称轴的方向，对称轴长度就是特征值的倒数开方。如果由特征值为负数，即矩阵非正定，则二次曲面为马鞍面，也很清楚。

# 19.相似矩阵和约当标准型
矩阵 $A,B\in \mathbb R^{n\times n}$ 若存在可逆矩阵M使 $A=M^{-1}BM$，则A和B相似。在矩阵对角化中我提过一嘴，A和对角化后的矩阵就是相似的。当然我们可以选择很多可逆矩阵，这样就得到很多相似的矩阵，它们的共同点就是特征值相同：

$$\begin{align}Ax&=\lambda x,A=M^{-1}BM\\
&\Rightarrow M^{-1}BMx=\lambda x\\
&\Rightarrow MM^{-1}BMx=\lambda Mx \\
&\Rightarrow B(Mx)=\lambda (Mx)\end{align}$$

A和B的特征值相同，但是特征向量不同，相差一个线性变换M。但是让我头皮发麻的重点其实是：它们表示同样的线性变换，只是向量表达的基不一样。

但是特征值相同的矩阵不一定是相似的，因为存在重根的情况，矩阵可能不能对角化。比如同样的两个特征值都为4的二阶矩阵

$$A=\left[\begin{matrix}4&0\\0&4\end{matrix}\right],B=\left[\begin{matrix}4&1\\0&4\end{matrix}\right]$$

矩阵B只有一个线性无关的特征向量，所以不能对角化成A的形式，矩阵A只和自己相似，而其他特征值等于（4，4）的矩阵都相似于B，B是其中最简单的形式，它最接近对角阵。这种不能对角化的矩阵的最简形式又叫约当标准型（Jordan form）。这样一来，所有能对角化的矩阵都相似于对角阵，不能对角化的矩阵相似于约当标准型。约当标准型就是除了对角线上有元素，就只有对角线上面一行有1或0，每多一个1特征向量就少一个。约当标准型可以分解成几个约当块（Jordan block）的形式

$$J_i=\left[\begin{matrix}\lambda_i&1&0&0
\\0&\lambda_i&\ddots &0 \\
0&0&\ddots&1\\
0&0&0&\lambda_i\end{matrix}\right],
J=\left[\begin{matrix}J_i&0&0&0
\\0&J_j&0&0 \\
0&0&\ddots&0\\
0&0&0&J_k\end{matrix}\right]$$

每个约当块对应一个特征向量，就是说同一个特征值可能有多个约当块，如果矩阵有n个不同的特征值则每个约当块都是1乘1的，最后的约当标准型就成了对角阵。这块说的比较简略，因为现在约当型已经不是线性代数中的重点了。

# 20.奇异值分解（singular value decomposition）
这大概是最后的大头儿了。

$$A_{m\times n}=U_{m\times m}\Sigma_{m\times n} V_{n\times n}^T\tag1$$

其中A是任意矩阵，UV是单位正交矩阵，中间是对角阵。这里面也有一点几何意义，我们考虑A的行空间和列空间，我们希望在行空间中找到一组单位正交基 $v_1,v_2,\cdots,v_r$，经过A的线性变换后在列空间中得到另一组正交基 $\sigma_1 u_1,\sigma_2 u_2,\cdots,\sigma_r u_r$，其中r是矩阵的秩，$u_i$ 是单位正交基，这个转换可以写成：

$$A\left[\begin{matrix}v_1&\cdots&v_r\end{matrix}\right]=\left[\begin{matrix}u_1&\cdots&u_r\end{matrix}\right]
\left[\begin{matrix}\sigma_1& & \\ &\ddots&\\& &\sigma_r\end{matrix}\right]$$

如果有零空间的话，我们也可以在零空间中找到n-r个单位正交基，将左边的V补充为完整的 $\mathbb R^n$ 中的基，这部分经过线性变换A会得到0，我们可以在对角阵中用0补充，而矩阵U用左零空间中的基补充，这样我们就得到了 $\mathbb R^m$ 的完整基，于是我们得到了式(1)的形式。

$$A\left[\begin{matrix}v_1&\cdots&v_n\end{matrix}\right]=\left[\begin{matrix}u_1&\cdots&u_m\end{matrix}\right]
\left[\begin{array}{c|c}
\begin{matrix}\sigma_1&&\\&\ddots&\\&&\sigma_r\end{matrix}&0\\ \hline 0&0
\end{array}\right]$$

那么我们怎么求VU和 $\Sigma$ 呢。

$$A^TA=V\Sigma^TU^TU\Sigma V^T=V\left[\begin{matrix}\sigma_1^2&&\\&\sigma_2^2&\\ &&\ddots\end{matrix}\right]V^T \\
AA^T=U\Sigma V^TV\Sigma ^T U^T=U\left[\begin{matrix}\sigma_1^2&&\\&\sigma_2^2&\\ &&\ddots\end{matrix}\right]U^T$$

所以分别对 $A^TA,AA^T$ 矩阵对角化，然后 $\sigma$ 取对角阵中元素正的平方根。这里有一点要注意，实对称矩阵一定可以对角化（证明略，我不会），所以任意矩阵都可以用上述方法奇异值分解。

# 21.线性变换
我们到现在才提出线性变换的概念，实际上我在之前的文章中已经多次提及了矩阵对应着线性变换这一事实，这里完整讨论一下。首先线性变换是不依赖坐标系的，假设一个线性变换T将向量v变换为 $T(v)$ ，且满足以下两点，则称T是一个线性变换：

1. $T(c\boldsymbol v)=cT(\boldsymbol v)$
2. $T(\boldsymbol v+\boldsymbol w)=T(\boldsymbol v)+T(\boldsymbol w)$

所以线性变换是一种映射关系，比如二维平面上的投影就是一种映射，也是一种线性变换。它将一个向量映射为另一个向量 $T:\mathbb R^2 \rightarrow \mathbb R^2$，而且满足上述两条性质。而如果建立了坐标系，我们就可以通过一个矩阵表达这个线性变换，即 $T(v)=Av$。**每一个线性变换通过建立坐标系都可以用矩阵表达，而每一个矩阵都表示一个线性变换**。我们已经知道：

$$A(c\boldsymbol v)=cA\boldsymbol v \\
A(\boldsymbol v+\boldsymbol w)=A\boldsymbol v+A\boldsymbol w$$

所以矩阵乘以向量确实是一种线性变换。具体的，我们怎么把一个线性变换T表达为矩阵呢。

线性变换作用于整个空间会改变其中所有的向量，我们怎么知道它是如何作用于每一个向量的呢，需要求出所有的变换结果吗？答案是不需要。根据线性变换性质，我们只要确定空间中的所有基在线性变换下的结果就行了，这样其他向量的变换结果都可以用基的变换结果表示：

$$v=c_1v_1+c_2v_2+\cdots+c_nv_n\\
T(v)=c_1T(v_1)+c_2T(v_2)+\cdots+c_nT(v_n)$$

只要选定了基，我们就可以把每个向量写成基的线性组合的形式，系数就是该向量的坐标，显然改变基向量，坐标也会随之改变（这里选择的不一定是标准正交基）。

对于线性变换 $T:\mathbb R^n \rightarrow \mathbb R^m$，先选择输入空间的一组基 $v_1,\cdots,v_n$，输出空间的一组基 $w_1,\cdots,w_m$，注意我们可以在输入输出空间选择不一样的基。然后矩阵A的每一列就对应下列各式的系数：

$$T(v_1)=a_{11}w_1+a_{21}w_2+\cdots+a_{m1}w_m\\
T(v_2)=a_{12}w_1+a_{22}w_2+\cdots+a_{m2}w_m\\ \vdots$$

然后输入输出空间的向量都用各自的基表示，记为v和w，则有 $Av=w$。我们可以验证对于基向量，该变换是正确的，所以对于空间中所有向量该形式都是正确的。

下面是几个有意思的知识点，写到一起了

1. 在可以选择的各种基向量中，特征向量是比较好的，这样矩阵A就是对角阵的形式。参考矩阵对角化和奇异值分解。
2. 矩阵的逆对应线性变换的逆变换，矩阵相乘就是先后做两个线性变换。
3. 如果改变空间中基向量，则向量坐标的变化也是一种线性变换，该变换对应的矩阵的每一列是新的基在老的基下的坐标。
4. 对于 $\mathbb R^n \rightarrow \mathbb R^n$ 的线性变换，选择不同的基得到的不同矩阵是相似的。
