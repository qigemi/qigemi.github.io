---
layout: post
---

# 0.序
有人推荐了b站上的一个up主，会发一些有意思的和数学相关的视频，我在其中发现了[线性代数系列](https://www.bilibili.com/video/av5977466)。
里面的视频做得很形象，是从线性变换的角度出发，描述了矩阵如何将向量变换到另一个位置。
然后通过变换中的一些特殊情形，引出列空间，零空间，行列式等概念。最后还提了一嘴其他类型的向量，如函数，也引出了基函数和傅立叶分解等。
总之整个系列经常让我发出“aha”的感叹，如果早点看这视频就好了。

后来我又开始看MIT[公开课](http://open.163.com/special/opencourse/daishu.html)，这也是很经典的课，里面的老头讲课有点随性，但是能让人更加理解线性代数的内容，和我本科时听的线代完全不一样。
公开课更系统一些，b站那个视频系列只是通过可视化讲了一部分。下面主要总结公开课中的内容。

写完全部之后回看，感觉自己写的有些零散，逻辑不连贯，而且这门课的精髓其实在于各种各样简单的例子，我嫌麻烦丢弃了很多例子，只总结了一些结论，各位看起来可能没有恍然大悟，相见恨晚的感觉。另外课程中不断出现的四个子空间的图形化表示，我都没有贴出图片来。所以整个系列博客最后的成果可能和我的初衷有些距离，我最开始是希望大家看了我的博客就不用再看视频了，现在看来相去甚远。可能这个系列的作用只在于提炼和记录，大家可以先看看本文，有哪些不懂的，感兴趣的再去看对应课程。希望对大家有所帮助。

第一部分，先介绍消元法。

---

# 1.矩阵和消元（elimination）
这里是用线性方程组引出矩阵的概念的，线性方程组可以写成 $Ax=b$ 的形式，如果我们考虑矩阵的行，每行表示方程组中的一个方程，每个方程可以看作一条直线，则方程组的解就是这些直线的交点的坐标。  
如果考虑矩阵的列，即方程组可以写成下列形式（假设是两个未知数），其中 $a_1,a_2$ 都是列向量：

$$\left[ \begin{matrix} \boldsymbol a_1 & \boldsymbol a_2 \end{matrix} \right]
\left[\begin{matrix} x_1 \\ x_2\end{matrix}\right]=b\\
\boldsymbol a_1x_1 + \boldsymbol a_2x_2 = b\tag{1}$$

则等价于求矩阵列向量的线性组合使其等于b。（后面会讲到：这里就出现了方程组可能无解的情况，因为如果列向量不是线性无关的，则b可能无法用列向量表示出来，此时方程组无解。）

以上是对方程组的几何意义和代数意义的解释。下面介绍一种矩阵形式下求解方程组的方法： **高斯消元法** 。  
我们知道对方程组做初等行变换不会改变解，高斯消元法就是对增广矩阵做初等行变换，使矩阵A变成上三角的形式，这样可以从最后一个方程开始，每次只求解一个一元一次方程，比较简单。这里举个例子

$$\left[ \begin{matrix} 1 & 2 & 1 \\ 3&8&1 \\ 0&4&1 \end{matrix} \right]
\left[\begin{matrix} x_1 \\ x_2\\x_3\end{matrix}\right]=
\left[\begin{matrix} 2\\12 \\ 2\end{matrix}\right]$$

经过消元得到

$$\left[ \begin{matrix}
\boldsymbol{1} & 2 & 1 \\
 0 & \boldsymbol 2 & -2 \\
  0 & 0 & \boldsymbol 5 \end{matrix}\right]
\left[\begin{matrix} x_1 \\ x_2 \\ x_3 \end{matrix}\right]=
\left[\begin{matrix} 2 \\ 6 \\ -10 \end{matrix}\right]$$

从最后一行解得 $x_3=-2$，带入第二行得 $x_2=1$，最后 $x_1=2$。消元后的上三角矩阵记为U（upper triangular），其对角线上的元素称为主元（pivot），如果消元过程中对角线主元位置上出现0，则通过换行将下面行中该位置不是0的行换上来。

**如何将消元的过程用矩阵表示？** 我们知道矩阵右乘向量表示矩阵列的线性组合，如式（1）所示，而左乘向量则得到行的线性组合。消元是对行做线性组合的过程，所以消元可以表示为矩阵左乘一系列矩阵的形式：

$$\left[ \begin{matrix} 1 & 0 & 0 \\ 0 &  1 & 0 \\ 0 & -2 & 1 \end{matrix}\right]
\left[\begin{matrix} 1 & 0 & 0 \\ -3 &  1 & 0 \\ 0 & 0 & 1 \end{matrix}\right]
\left[\begin{matrix} 1 & 2 & 1 \\ 3 & 8 & 1 \\ 0 & 4 & 1 \end{matrix}\right]=
\left[\begin{matrix} 1 & 2 & 1 \\ 0 & 2 & -2 \\ 0 & 0 &  5 \end{matrix}\right]$$

将消元矩阵记为E，则上式可表示为：

$$E_{32}E_{21}A=U$$

这些变换矩阵E都是可逆的，等式两边乘以它们的逆可以得到著名的LU分解：

$$A=E^{-1}U=LU\\
\left[\begin{matrix} 1 & 2 & 1 \\ 3 & 8 & 1 \\ 0 & 4 & 1 \end{matrix}\right]=
\left[\begin{matrix} 1 & 0 & 0 \\ 3 & 1 & 0 \\ 0 & 2 & 1 \end{matrix}\right]
\left[\begin{matrix} 1 & 2 & 1 \\ 0 & 2 & -2 \\ 0 & 0 &  5 \end{matrix}\right]$$

注意 $E^{-1}$ 是一个下三角矩阵，而且其中的元素和消元乘数是对应的，对角线上的元素都是1，我们把它记为L（lower triangular）。有时我们也可以把U中的主元提取出来作为对角阵D（diagonal），则分解变为：

$$A=LDU$$

当然其他情况中还可能出现交换行的操作，即出现主元位置为0的情况，该操作对应的矩阵E又叫置换矩阵（permutation matrix，这里有一个群论的大坑，不知道什么时候填）。如果需要交换行，则需要增加一个所有置换矩阵的乘积P：

$$PA=LU$$

# 2.乘法和求逆
矩阵乘法我们都会计算，$AB$ 可以看成矩阵B的行的线性组合，或者矩阵A的列的线性组合，也可以用定义式或者分块计算。

矩阵逆是针对 **方阵** 的一个概念，定义为 $A^{-1}A=AA^{-1}=I$，不是所有方阵都有逆，比如列或行线性相关的方阵就不可逆。方阵不可逆，当且仅当存在 $x\neq 0$ 使得 $Ax=0$，即矩阵的列线性相关。  

**这里介绍一种通过消元求逆的方法**，高斯-约当消元法，比高斯消元多两步，  
（1） 回代使主元上方的所有元素等于0，  
（2） 使主元为1，这样消元结果就是单位阵。举个例子：

$$A=\left[\begin{matrix} 1 & 3 \\ 2 & 7 \end{matrix}\right]\quad
\left[A\ I\right]=\left[\begin{matrix} 1 & 3 & 1 & 0 \\ 2 & 7 & 0 & 1 \end{matrix}\right]$$

对增广矩阵做高斯-约当消元，可得：

$$\left[\begin{matrix} 1 & 0 & 7 & -3 \\ 0 & 1 & -2 & 1 \end{matrix}\right]$$

则右边的方阵就是A的逆。神奇吗？proof：由上节可知消元过程可以等价于左乘矩阵E，则：

$$E\left[A\ I\right]=\left[EA\ EI\right]=\left[I\ E\right]\\ \Rightarrow EA=I$$

# 3.LU分解，转置与置换
这节课主要推导了LU分解，计算了消元法的计算复杂度。对于矩阵 $A\in \mathbb{R}^{n\times n}$ ，在不考虑交换行的情况下，消元的复杂度为 $n^3/3$，在增广矩阵的情况下，对b的操作的复杂度为 $n^2$。

矩阵转置定义为 $A_{ij}^T=A_{ji}$，如果 $A^T=A$ 则A称为对称矩阵（symmetric matrix）。

到这里消元这部分就告一段落了，下面进入向量空间的部分。

---