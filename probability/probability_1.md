---
layout: post
---

# 1.概率模型

当我们想知道某种情况下，某些事情发生的概率时，我们要先建立一个概率模型，方便计算。总共分两步，确定采样空间和采样空间上计算概率的规则。

## 采样空间（sample space）

一个实验的所有可能结果构成的集合叫做采样空间。比如扔硬币，采样空间可以是（正面向上，反面向上），也可以是（正面向上且天气是晴天，正面向上且不是晴天，反面向上）。

我们可以添加无数多的条件来组成不同的结果，但是这些条件可能并没有对我们关心的实验结果造成影响。比如我们关心硬币的哪面朝上，那么我们可以忽略天气的影响，这样可以让模型更简单。

这也是建模的很常用的思想，毕竟现实中影响实验结果的因素有很多，我们不可能完全考虑，所以在建模时要斟酌一下哪些因素是主要的，那些可以忽略。

上述采样空间是有限集合，那么就有实验结果是无穷多的实验，比如在（0，1）中随机选一个实数，这种采样空间就是无限集合。

注意采样空间中的元素应该是 **互斥** 的，即一次实验只能发生一个结果，同时采样空间应该包含所有的结果。

有了采样空间，下一步我们就要计算其中元素对应的概率，来表示某个或某些实验结果发生的可能性。对于有限集合我们可以给每个元素一个概率，比如扔硬币正、反概率分别为0.5。但是无限集合我们无法这么做，因为直观上实验结果刚好是其中某一个元素的概率都是0。比如在（0，1）中刚好选中0.5的概率是0（或者说无穷小？），这对我们是没有意义的。所以我们要计算的是采样空间中的子集对应的概率。

采样空间中的子集即定义为 **事件（event）**。一个事件发生的概率就是我们认为实验结果落在这个子集中的可能性。我们可以根据自己的经验给出事件发生的概率，但是概率要满足几个公理：

1. 非负：$P(A)\geq 0$
2. 整个采样空间所对应的概率是1：$P(\Omega)=1$
3. 可加：$if\ A\cap B=\emptyset,then\ P(A\cup B)=P(A)+P(B)$

第三条可加性，可以推广到两个以上事件相加，实际上如果事件可以被明确的数清楚，则无穷多个相加也可以。

> 这里要注意可数，实数集上的数列就是不可数的，不能应用此定理。比如某个面积对应的概率不能表示成每个点的概率相加。连续模型的概率的意义和离散模型也有一些不同，比如连续模型上的概率等于1不表示一定发生，而是极大的可能发生，等于0也不表示一定不发生。

我们要求概率应该在0-1之间，但是公理中不需要明确的说明这一点，因为前两条隐含的要求了这一点。

## 概率计算规则

在采样空间上我们定义一套概率计算规则，常用的规则有：

1. 离散均匀分布规则：认为每个结果（注意是离散的）发生的概率一样
2. 连续均匀分布规则：比如采样空间是一个矩形，概率可以正比于事件的面积

这里只举了例子，似乎没有严格的定义。比如矩形中选中一个点的概率是0，因为点没有面积，直线也没有面积，这个和测度论有关，课程里可能不会讲这些。

总之这就是计算简单概率的大体方法，先找到所有的可能结果建立采样空间，然后确定采样空间中子集所对应的概率的计算规则，然后就可以计算该实验所有事件所对应的概率。好像什么都没说，但是这是一个通用的方法，可以说是后面很多概率计算的本质。

# 2.条件概率和贝叶斯法则

当我们确定了概率模型之后，就可以计算概率了，比如某个事件的概率。这时如果有人告诉了你关于实验的其他信息，则我们应该根据这些信息修改我们之前计算的概率，因为我们有了更多的信息和判断依据。

$P(A\vert B)$ 表示给定B发生的情况下，事件A发生的概率。这里B就是我们新的采样空间了。

我们规定 $P(B) \neq 0$ ，则

$$P(A\vert B)=\frac{P(A \cap B)}{P(B)}$$

如果B的概率等于0，则该条件概率是未定义的。

条件概率也是概率，只是采样空间不同，缩小为原来全集的子集B。上一节中的公理对条件概率也适用。

*例子*：

雷达监测飞机的例子，有飞机为事件A，没有为A'，雷达监测到飞机为事件B，没检测到是B'。已知 $P(B\vert A)=0.99$, $P(B'\vert A)=0.01$, $P(B\vert A')=0.1$, $P(B'\vert A')=0.9$。
可以看出雷达识别正确的概率是很高的，都在90%以上，但是如果雷达报了监测到目标，而且确实有飞机的概率是多少呢。此时还需要知道有飞机的先验是多少，即 $P(A)=0.05, P(A')=0.95$。

已知上面这些信息之后需要经过三步计算，

$$P(A \cap B)=P(A)*P(B\vert A)=0.0495 \\
P(B)=P(A)P(B\vert A) + P(A')P(B\vert A')=0.1445 \\
P(A\vert B)=0.34$$

实际上 $P(A\vert B)$ 只有0.34。为什么会和直觉相差这么多呢，因为有飞机的概率很小，大部分时间是没有飞机的，而此时雷达有10%的误检率，所以在雷达有信号的所有情况中，其实误检占了很大部分。

再来看上面三个步骤，对应一般概率计算中的三个重要的公式。第一个式子类似链式法则，可以求多个事件共同发生的概率，即条件概率的连乘。

第二个式子是全概率公式，假设将全空间分成几个小空间A1,A2,A3等等，则 B 发生的概率可以写为

$$P(B)=P(A1)P(B\vert A1)\\+P(A2)P(B\vert A2)\\+P(A3)P(B\vert A3)$$

注意，因为$P(A1)+P(A2)+P(A3)=1$，所以上式也可以看成是一种**加权平均**。全概率公式让我们可以将问题分解为多个子问题进行求解，可能可以简化求解过程。

第三个是贝叶斯公式，展开写：

$$P(A\vert B)=\frac{P(A \cap B)}{P(B)}=\frac{P(A)P(B\vert A)}{P(A)P(B\vert A) + P(A')P(B\vert A')}$$

可以看到我们已知的都是在A发生条件下，B发生的概率，比如有飞机时雷达报警的概率是多少？没有飞机时又是多少？这是一种**因果**关系。
而经过贝叶斯公式我们得到了B发生情况下A发生的概率，雷达报警了有飞机的概率是多少？没有飞机概率是多少？这是有“果”的情况下，我们**推断**“因”的过程。

$$A_i \xrightarrow[P(B\vert A_i)]{cause-effect} B \\ A_i \xleftarrow[P(A_i\vert B)]{inference} B$$

在实际应用中，我们可以通过观测结果，和贝叶斯公式来推断引起结果的原因。这是很多状态估计的基本方法。

总结：
1. 条件概率 $P(A\vert B)=\frac{P(A \cap B)}{P(B)}$
2. 乘法规则 $P(A \cap B)=P(A\vert B)P(B)=P(B\vert A)P(A)$
3. 全概率公式
4. 贝叶斯定理

# 3.独立（independence）

两个时间相互独立，即一件事的发生，不会改变你对另一件事发生与否的概率估计。

$$P(B\vert A)=P(B) \tag{1} $$
$$P(A \cap B)=P(A\vert B)P(B)=P(A)P(B) \tag{2} $$

2式由1式推导而来，但是2式更常用，因为1式需要在P（A）大于零的时候才成立，而2式任何时候都成立。
当且仅当（2）式成立时，我们称事件A和B独立。当然这是严谨的证明，我们如果能确定事件A的发生对B没有任何影响，就可以认为他们独立。

对于条件概率的情况：

$$P(A \cap B\vert C)=P(A\vert C)P(B\vert C)$$

**注意如果AB独立，那么在C发生的条件下AB不一定独立。反之，如果在C的条件下独立，A和B也不一定独立。**

对于多个事件，如果有

$$P(A1 \cap A2 \cap A3)=P(A1)P(A2)P(A3)$$

则称它们相互独立。**注意如果几个事件两两独立，他们不一定相互独立。**

这里有一个例子。扔两次骰子，A:第一次是正面，P=0.5；B:第二次是正面，P=0.5；C:两次结果相同，P=0.5.

其中，ABC两两独立，因为 $P(AB)=P(BC)=P(AC)=0.25$。但是ABC并不是相互独立，因为 $P(C\vert A \cap B)=1 \neq P(C)$ 。

还有另一个著名的例子：一个家庭有两个孩子，已知其中一个是男孩的情况下，另一个是女孩的概率是多少？答案是2/3，因为其中一个是男孩有三种情况：（男，男）（男，女）（女，男）。另一个是女孩占其中的2/3。

我第一次看这个问题时认为答案是0.5，并且觉得条件概率这种解释太牵强了，是在强行应用，后来才明白是我理解错了题目表达的意思，它并没有说第一个孩子是男孩，而是其中有一个是男孩，这两种表述所表示的条件确实不同。

抛开我的理解错误，这个题目真正要我们注意的地方是，同一个问题，如果加上不同的前提，答案就会不同。比如，如果已知这个家庭会一直生孩子直到生出男孩，则另一个孩子是女孩的概率就是1。
**这就需要我们在解决实际问题的时候，一定要小心的提出假设，哪些条件是可忽略的（分布独立的就可以忽略），哪些必须考虑。** 当然也许没有绝对的答案，只能说在误差和计算量可接受的范围内，尽量调整算法，有点经验在里面。

# 4. counting

这一节主要介绍了一些比较初级的排列组合知识。
在可数事件方面还有很多更复杂的问题，这里没有介绍。

# 5.离散随机变量

## 5.1 定义

首先定义随机变量，**随机变量是一个从采样空间到实数的函数映射。**

比如定义一个实验是从一群学生中随机选一个，我们取选到的学生的身高作为输出，则得到了一个从采样空间到实数的映射，即随机变量，可以记为H。

同一个实验可以定义不同的随机变量，比如还是上述实验，我们可以取每个学生的体重，则会生成另一个随机变量。

随机变量的函数也是一个随机变量，因为它同样是采样空间到实数的映射。

如果随机变量映射到比如整数的集合，则称为离散随机变量，映射到实数集则称为连续随机变量。首先讨论离散随机变量。

这里一定要区分随机变量 $X$ 和随机变量的取值 $x$ 两个概念。随机变量是一个函数，输入一个实验结果它就输出一个实数。而随机变量的值就是一个实数。

## 5.2 概率分布函数（probability mass function PMF）

既然随机变量X可以取不同的值，我们就先讨论X取不同值时的概率。

$$p_X(x)=P(X=x)\\=P(\{ \omega \in \Omega \ s.t. \ X(\omega)=x  \})$$

有两条性质

$$p_X(x) \ge 0 \ \ \  \sum_x p_X(x)=1$$

这其实与求事件的概率差不多，就是定义所有使 $X$ 输出 $x$ 的实验结果构成一个事件，求这个事件的概率。

## 5.3 随机变量的期望和方差

期望可以认为是一种平均，由概率加权的平均。

$$E[X]=\sum_x p_X(x)x$$

如果我们知道X的概率分布，又知道 $Y=g(X)$，如何计算 Y 的期望。根据定义我们需要计算

$$E[Y]=\sum_y p_Y(y)y$$

但是这样我们需要计算Y的概率分布，另一种简单的方法是

$$E[Y]=\sum_x p_X(x)g(x)$$

注意，一般情况下 $E[g(X)] \neq g(E[X])$，但是线性关系是成立的，设 $\alpha, \beta$ 为常数

$$E[\alpha]=\alpha \\ E[\alpha X]=\alpha E[X] \\ E[\alpha X+\beta]=\alpha E[X]+\beta$$

对于**方差**也有一个重要的公式

$$var(X)=E[(X-E[X])^2]=E[X^2]-(E[X])^2$$

性质：

$$var(X) \geq 0 \\ var(\alpha X)=\alpha ^2var(X)$$

## 5.4 条件概率分布函数

条件概率也是正常概率，只是采样空间不同。

$$p_{X\vert A}(x)=P(X=x\vert A) \\ E[X\vert A]=\sum_x xp_{X\vert A}(x)$$

所有期望和方差的性质对条件概率分布都适用。

然后我们来看一种概率分布，几何概率分布函数，它的条件概率分布有一个有趣的性质。
几何分布是如下形式。假设扔硬币正面向上的概率为p，则连续扔直到出现正面向上所需要的次数为随机变量X。

$$p_X(k)=(1-p)^{k-1}p \\ p_{X-2\vert X>2}(k)=p_X(k)$$

第二个式子这样理解，如果我们先扔了两次，都是反面，那么 *后面继续扔* 和 *重新开始扔* 的概率分布应该一样。
也就是已知X>2的情况下，（X-2）这个随机变量的概率分布和从零开始是一样的。

然后我们利用这个性质来计算几何概率分布的期望。利用全概率公式，两边同时取期望：

$$p_X(x)=P(A_1)p_{X\vert A_1}(x)+ \cdots +P(A_n)p_{X\vert A_n}(x) \\
E[X]=P(A_1)E[X\vert A_1]+ \cdots +P(A_n)E[X\vert A_n]$$

对于几何分布，设事件A1：X=1，事件A2：X>1。

$$\begin{align}
E[X]&=P(X=1)E[X\vert X=1]+P(X>1)E[X\vert X>1]\\
    &=p \cdot 1 + (1-p) \cdot (E[X]+1)
\end{align}$$

解方程得 $E[X]=1/p$。

## 5.5 联合概率分布

$$p_{X,Y}(x,y)=P(X=x \ and \ Y=y)$$

有时我们想知道两个随机变量之间有什么关系，比如身高和体重，他们之间应该有某种关系。
联合概率分布就可以体现这种关系。有以下式子：

$$\sum_x \sum_y p_{X,Y}(x,y)=1 \\
p_X(x)=\sum_y p_{X,Y}(x,y) \\
p_{X\vert Y}(x\vert y)=P(X=x\vert Y=y)=\frac{p_{X,Y}(x,y)}{p_Y(y)}$$

第二个式子称为边缘概率，有点类似全概率公式，比如要求X=x时的概率，就将所有X=x的情况都加起来。
第三个式子是条件概率，给定Y=y时X的概率分布。同理可以推广到多个变量的联合概率分布。
另外，如果两个变量独立，则有，

$$p_{X,Y}(x,y)=p_X(x)p_Y(y)$$

同样也有给定条件下的独立。然后讨论一下期望，有一些关系：

$$E[X+Y+Z]=E[X]+E[Y]+E[Z]$$

如果XY相互独立：

$$E[XY]=E[X]E[Y] \\
E[g(X)h(Y)]=E[g(X)]E[h(Y)]$$

这里要注意，如果XY独立，则他们的函数 $g(X),h(Y)$ 也相互独立。

方差：如果XY独立则：

$$var(X+Y)=var(X)+var(Y)$$

然后是一个有意思的例子。有n个人，每个人有一顶帽子，所有的帽子混到一起，每人随机拿一个，定义随机变量X为拿到自己帽子的人数，求X的期望。

首先我们给每个人定义一个随机变量Xi，1表示拿到了自己的帽子，0表示不是自己的帽子。则：

$$X=X_1+X_2+ \cdots +X_n \\
P(X_i)=\frac{1}{n} \\
E[X_i]=1 \cdot \frac{1}{n} + 0 \cdot (1-\frac{1}{n})=\frac{1}{n}\\
E[X]=\sum E[X_i]=1$$

虽然Xi相互之间并不独立，但是期望的线性关系是一直成立的，所以最后一个式子成立。所以不管有多少人，拿到自己帽子的人数期望总是1。

那么方差怎么算呢。

$$var(X)=E[X^2]-E[X]^2$$

第二项等于1，我们来算第一项

$$X^2=(\sum X_i) ^2=\sum(X_i^2)+\sum_{i \neq j}X_iX_j$$

再分别算这两项的期望，因为Xi值取0，1，所以算期望的时候我们只要算他们取1的概率就行了，

$$E[X_i^2]=\frac{1}{n} \\
E[X_iX_j]=P(X_i=1,X_j=1)=\frac{1}{n}\frac{1}{n-1}$$

带回去得：

$$E[X^2]=n \cdot \frac{1}{n}+n(n-1)\cdot \frac{1}{n(n-1)}=2 \\
var(X)= 2- 1 = 1$$

无论n为多少，X的方差也都等于1！

# 6.连续随机变量

取值为连续实数的随机变量为连续随机变量。

我们用概率分布函数描述离散随机变量取不同值时的概率。但是对于连续随机变量，每个取值对应的概率值是无限趋近于0的，所以我们用一种积分的思想来表示连续随机变量的概率分布。
虽然每个点的概率都是0，但是在一个区间上积分的结果不是0，这就可以表示随机变量在不同取值附近的概率分布情况。

我们用概率密度函数 $f(x)$ 表示连续随机变量的概率分布情况。probability density function。

$$P(a \le X\le b)=\int_a^bf(x)dx\\
f(x) \ge 0 \\
\int_{- \infty }^\infty f(x)dx=1$$

由1式我们可以推出 $P(X=a)=0$，也就是连续随机变量取任意值的概率都是0，所以 $f(x)$ 并不表示X取不同值时的概率大小，它可能大于1，但是它在实数域上的积分等于1。

还可以这样思考概率密度函数，对于无穷小的 $\delta$ 有：

$$P(a \le X\le a+\delta)=\int_a^{a+\delta}f(x)dx=f(x)\delta$$

意思是，在a附近无穷小的区间内，每单位长度对应的概率值为 f(x)。由上面几个公式可以看出，连续随机变量的概率都和区间相关，每个区间对应一个概率值。这样我们就可以把离散随机变量中的各种知识直接应用过来了。

$$E[X]=\int_{- \infty }^\infty xf(x)dx \\
E[g(X)]=\int_{- \infty }^\infty g(x)f(x)dx \\
var(X)=\int_{- \infty }^\infty (x-E[X])^2f(x)dx\\
=E[X^2]-E[X]^2$$

我们只要把求和变成积分就行了。

## 累积分布函数

这时有一个问题，有没有一种形式可以统一离散和连续随机变量呢？可以用累积分布函数（cumulative distribution function）。

对于连续随机变量：

$$F(x)=P(X\le x)=\int_{- \infty }^x f(t)dt$$

离散变量：

$$F(x)=P(X\le x)=\sum_{k\le x}p(k)$$

当一个变量既可能取连续值，又可能取离散值的时候，用CDF描述就很合适了。

## 6.1 高斯分布 Gaussian（正态分布 normal）

高斯分布是很重要的一个分布，因为数量足够多的任意独立分布相加，其结果将近似于高斯分布。所以在实际实验中，有多种因素影响的时候，我们一般会假设噪声是高斯分布的。

正态分布 $N(\mu,\sigma^2)$:

$$f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

我们应该都清楚，该函数是左右对称的，对称轴在均值处，而方差决定了曲线的胖瘦，也就是分散程度。

正态分布随机变量的线性函数也是正态分布的。

由于 f(x) 的积分没有闭式解，所以我们很难计算概率，比如 $P(X \le k)$。所以一般我们计算类似概率的时候，先把任意正态分布转换成标准正态分布，再去查标准正态分布的累计概率函数表，由此得到任意正态分布的累积概率。

## 6.2 多维连续随机变量

联合概率密度函数

$$P((X,Y)\in S)=\int \int_S f_{X,Y}(x,y)dxdy$$

边缘化，我们计算在 X=x 的无穷小区间内对应的概率，

$$f_X(x) \cdot \delta=\int_{- \infty }^{\infty}\int_{x}^{x+\delta} f(\hat x,y)d\hat xdy\\
=\int_{- \infty }^{\infty} \delta \cdot f(x,y)dy\\
\Rightarrow f_X(x)=\int_{- \infty }^{\infty} f(x,y)dy$$

最终的形式和离散随机变量的全概率公式相同。

我们称X和Y相互独立，如果对所有的x，y有：

$$f_{X,Y}(x,y)=f_X(x)f_Y(y)$$

然后是条件概率密度函数：

$$f_{X\vert Y}(x,y)=\frac{f_{X,Y}(x,y)}{f_Y(y)} \ if\ f_Y(y)>0$$

最后是一个有趣的例子。在等间距的平行线中（间距为d），扔一根长度为 l 的针（l < d)。因为针的长度小于间距，所以每次只能和一条线相交。问针与线相交的概率。

这里将用到最基本的四步法：1.建立采样空间；2.给空间中的样本赋值概率；3.确定我们要计算概率的空间；4.计算概率。

首先第一步，我们要确定的是针的位置，所以采样空间就是针的位置，那么如何表示位置呢。我们选择两个变量，一是针的中点的位置，用该点到最近直线的距离 X 表示, $X\in [0, d/2]$；二是针的角度 $\Theta\in [0,\pi /2]$。

第二步，计算每种情况的概率。因为位置和角度是独立的，所以可以分别计算概率然后相乘。然后又因为这两个随机变量都是均匀分布的，所以最后的概率可以表示为：

$$f_{X, \Theta}(x, \theta)=f_X(x)f_{\Theta}(\theta)=(2/d)\cdot (2/\pi)$$

第三步确定我们要计算的空间，针与直线相交，需要满足 $X\leq (l/2)sin\Theta$。

最后经过一些推导可得最终的概率为 $\frac{2l}{\pi d}$。

结果中有 $\pi$，这意味着我们可以用这个实验来估计圆周率！重复的做这个实验，通过统计概率就可以反算出圆周率的值。（另外，一般在每个和圆相关的实验中，我们都能得到和圆周率有关的结果，这可能和三角函数的积分有关。）

还有一个启发是，如果一个积分特别复杂，无法计算，我们可能可以通过多次随机实验的方式来估计。这称为蒙特卡洛方法。

## 6.3 连续随机变量贝叶斯法则

概率的一大作用就是帮助我们做推断，而根据观测实现推断的最好工具就是贝叶斯法则。这一点我们在离散情况中已经学到了。
连续随机变量的贝叶斯法则在形式上和离散情况一样。具体的例子，比如我们想测电流，但是电流表是有误差的，假设这个误差服从高斯分布，那么我们怎么从测量值估计真实值，这就可以应用贝叶斯法则。

这时就出现了离散和连续混合的情况，比如X是离散随机变量，取值0，1，Y是连续随机变量，它等于X的值加上一个高斯噪声。如何根据Y的值推断X呢。这种情况的贝叶斯公式如下推导：

$$P(X=x,y\leq Y\leq y+\delta)=P(X=x)P(y\leq Y\leq y+\delta\vert X=x)\\
=P(y\leq Y\leq y+\delta)P(X=x\vert y\leq Y\leq y+\delta)\\
\Rightarrow P_X(x)f_{Y\vert X}(y\vert x)\cdot \delta =f_Y(y)\cdot \delta \cdot P_{X\vert Y}(x\vert y)$$

两边的 $\delta$ 可以消掉。公式中P表示概率分布函数，f表示概率密度函数，它们和变量的形式是对应的。

# 7. derived distribution（导出分布）

什么是导出分布呢，我不知道这个翻译对不对，英文是derived distribution，我就先叫它导出分布吧。就是已知X的分布，求X的函数的分布。

假设 $Y=g(X)$ ，已知X的分布，如何求Y的分布呢？对于离散的情况，我们可以找到令Y=y的所有X值，然后把它们对应的概率相加：

$$P_Y(y)=\sum_{x:g(x)=y}P_X(x)$$

对于连续的情况呢？由于每个点的概率都等于0，我们不能用类似的方法。但是可以用累积分布函数来计算：

$$F_Y(y)=P(Y\leq y)=P(g(X)\leq y)=P(X\leq g^{-1}(y))$$

当然这个只能用于g是单调递增函数（如果是单调递减函数，则最后一个式子改为大于等于号。但是不是单调函数则不行）。然后对Y的累积概率函数求导就可以得到Y的概率密度函数了。

还有更通用的方法，就是利用微积分。这需要g是严格单调函数。

对于 $Y=g(X)$ ，事件 $x\leq X \leq x+\delta$ 等价于事件 $g(x)\leq Y \leq g(x+\delta)$，近似于 $g(x)\leq Y\leq g(x) +\delta \vert (dg/dx)(x)\vert$ 也就是相同概率，X，Y对应的无穷小区间长度不一样，有一个比例。具体关系为：

$$f(x)\delta =f(y)\delta \vert \frac{dg}{dx}(x)\vert $$

（据说可以从均匀分布生成任意分布，可能就是利用这个原理。比如我要生成一个奇怪的分布，只要有概率密度函数，就可以通过生成均匀分布的随机数，然后经过函数变换变成我们想要的分布。）

# 8.协方差

两个独立的高斯分布的联合概率密度函数为：

$$f_{X,Y}(x,y)=\frac{1}{2\pi \sigma_x \sigma_y}exp\{-\frac{(x-\mu_x)^2}{2\sigma_x^2}-\frac{(y-\mu_y)^2}{2\sigma_y^2}\}$$

我们思考这个概率密度函数的形状，用等高线的方式，概率密度函数相等的地方需要满足指数部分相等，而指数部分是一个椭圆方程，所以等高线是一组椭圆形，长短轴方向和xy轴方向重合。

如果XY线性相关，则椭圆的长短轴就不再沿着坐标轴的方向。这一点后面再说。

两个独立的高斯分布相加，$W=X+Y$, 则 $\mu_W=\mu_X +\mu_Y, \sigma_W^2=\sigma_X^2+\sigma_Y^2$。如果XY相关则均值和方差的形式也要有所变化。

协方差定义为：

$$cov(X,Y)=E[(X-E[X])(Y-E[Y])]\\=E[XY]-E[X]E[Y]$$

可以这样理解，协方差大于0意味着，当X比较大时，Y也有很高的概率比较大，且协方差越大，意味着这种相关性越强。
协方差小于0则是X比较大时，Y更可能偏小，同样协方差的绝对值越大，这种相关性就越明显。

**如果两个变量独立，则协方差为0。反之不成立。因为协方差是描述线性相关性的，协方差为0表示没有线性关系，但是有可能存在非线性关系，使两个变量并不独立。**

对于多个随机变量的和，其方差表示为：

$$var(\sum X_i)=\sum var(X_i)+\sum_{i\neq j}cov(X_i,X_j)$$

因为变量的量纲不同，可能导致不同的随机变量不能用协方差比较其相关性的大小，所以有无量纲的协方差形式，称为相关系数：

$$\rho =E[\frac{(X-E[X])}{\sigma_X}\frac{(Y-E[Y])}{\sigma_Y}]\\
=\frac{cov(X,Y)}{\sigma_X \sigma_Y}\\
-1\leq \rho \leq 1\\
\vert \rho \vert =1 \ \Rightarrow \ (X-E[X])=c(Y-E[Y])$$

相关系数的取值范围在-1到1之间，如果其绝对值等于1，意味着XY有严格的线性关系，给出一个X就有一个确定的Y对应。

# 9.迭代期望

首先我们再讨论一下条件期望。如果给定 $Y=y$，则 $E(X\vert Y=y)=\sum xp_{X\vert Y}(x\vert y)$。如果使连续随机变量则是积分的形式。

一个例子，将一根木棍第一次随机折断于y处，再将0-y的部分随机折断于x处，则 $E(X\vert Y=y)=y/2$。这是一个实数。

但如果我们不知道Y取何值，则 $E[X\vert Y]=Y/2$，这是一个随机变量，也就是说该条件期望的取值依赖于Y的取值，是随机变量Y的函数。

如果条件期望是随机变量，那么它也应该有一个期望（禁止套娃）。

$$E[E[X\vert Y]]=?$$

首先我们确定 $E[X\vert Y]=g(Y)$，则

$$\begin{align}
E[E[X\vert Y]]&=E[g(Y)]\\
&=\sum_y g(y)p_Y(y)\\
&=\sum_y E[X\vert Y=y]p_Y(y)\\
&=E[X]
\end{align}$$

我们注意到最后一个等式是全概率公式。

当然还有方差：

$$var(X\vert Y=y)=E[(X-E[X\vert Y=y])^2\vert Y=y]$$

这个就有点乱了。但是我们要知道 $var(X\vert Y)$ 同样是随机变量Y的函数。然后下面一个等式叫全方差公式：

$$var(X)=E[var(X\vert Y)]+var(E[X\vert Y])$$

这个证明看起来很爽，用到前面的知识：

$$var(X)=E[X^2]-(E[X])^2\\
var(X\vert Y)=E[X^2\vert Y]-(E[X\vert Y])^2\\
E[var(X\vert Y)]=E[X^2]-E[(E[X\vert Y])^2]\\
var(E[X\vert Y])=E[(E[X\vert Y])^2]-(E[X^2])$$

然后将后两个式子相加就行了。通过这个推导我们可以进一步熟悉和理解这些符号表示的意思。

对于全方差公式我们还可以这样理解。假设X表示每个学生的成绩，Y取值1或2，表示将学生分成两组，第一组10个人，第二组20个人。
两组学生的平均成绩分别为：

$$E[X\vert Y=1]=90,E[X\vert Y=2]=60$$

则 $E[X\vert Y]$ 就是一个随机变量，并且有：

$$E[E[X\vert Y]]=1/3 \cdot 90 + 2/3 \cdot 60 = 70 = E[X]\\
var(E[X\vert Y])=1/3\cdot (90-70)^2+2/3 \cdot (60-70)^2=200$$

再假设每组方差为：

$$var(X\vert Y=1)=10,var(X\vert Y=2)=20$$

则 $var(X\vert Y)$ 就是一个随机变量，并且有：

$$E[var(X\vert Y)]=1/3 \cdot 10+2/3 \cdot 20=50/3\\
var(X)=E[var(X\vert Y)]+var(E[X\vert Y])\\
=50/3 + 200$$

**最后X的方差分为两部分，第一部分是组内的方差，第二部分是组间的方差。**

# 结语

到这里为止，基本的概率概念就介绍完了。下一部分，我们将使用这些知识，介绍一些特殊的分布，以及在实际中如何推断（inference）。
