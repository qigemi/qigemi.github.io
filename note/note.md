---
layout: post
---

# 2019.4.25
-------
**PointNet**  
对三维点云的分类和分割网络，文章讨论了输入点顺序改变时网络性能的不变性，没看懂。
1. 输入$n\times3$矩阵，n为点数，3为x,y,z坐标，经过一个变换矩阵（由T-net生成）将点云变换到相对标准的位置，避免整体transform的影响。
2. 变换后经过多层感知机（MLP）分别对每个点进行处理，生成$n\times64$ feature，再对feature做（T-net）transform。
3. 对上述feature再做MLP，max pooling，FC等操作得到全局特征向量，由此可做分类。
4. 将该向量和$n\times64$ feature concatenate,经过MLP得到每个点的分类。

---
**FoldNet**  
基于pointnet的改进，用encoder-decoder结构做无监督学习，可以生成一个点云的特征向量，还可以通过该向量将一个点云变换为另一个。  
其中对pooling方法也有改进，他不是全局pooling，而是利用最近邻图（NNG）对某点附近的几个点做pooling（max or average）。

---
**On-the-Fly Adaptation of Regression Forests for Online Camera Relocalisation**  
一种在线训练回归森林做相机重定位的方法。  
related works：  
1. 图像匹配的方法，对图像生成描述子（词袋或其他向量描述子）  
2. key points匹配，生成的3D points和当前图像的2D points 匹配（RANSAC，Kabsch等）  
3. 其他方法，如ORB-SLAM结合上述两种方法，用神经网络直接回归6自由度pose，在当前位置附近生成3D points，将其和全局3D points匹配。  
4. 回归森林方法，和词袋类似，需要预先训练一棵回归树，应用时用来做3D-2D匹配。

本文的改进是在普遍场景中训练回归森林，然后去掉叶子，应用时online训练叶子。

# 2019.4.29
---
**NM-Net: Mining Reliable Neighbors for Robust Feature Correspondences**  
 一种判断对应点集中内点和外点的方法。 首先根据局部信息找到一些一致性好的相邻点，然后把邻接图输入神经网络（类似FoldNet），输出每个对应点是否是内点。

 ---
**Optimized Contrast Enhancements to Improve Robustness of Visual Tracking in a SLAM Relocalisation Context**  
<I>本文的直觉是HDR图像中提取的特征点在光照变化时比较稳定，所以通过建立SDR图像不同对比度增强的金字塔，模拟HDR图像的信息。</I>保存多层图像，每层根据互信息做不同的对比度增强，在多层图像中提取特征点。  
related works：
1. SuperFast：用类似反馈的方法动态优化Fast中的亮度阈值，图像每个区域可能有不同的阈值。
2. 输入HDR图像做色调映射，再提取特征点。对SDR图像主要是改变曝光时间调整对比度。
3. 对于直接法SLAM一般是在相邻帧之间做灰度放射变换，或者使用降维的深度学习特征，或者用互信息代替光度误差。  

本文中对每幅图像计算一个金字塔，第一层最大化变换（即对比度增强）后图像与参考图像（人为定义的亮度良好的图像）的互信息。第二层最大化参考图像与条件概率分布的互信息：

$$
\boldsymbol{u}^* = \mathop{\arg\min}_{\boldsymbol u} MI(I^{* }, f(I,\boldsymbol u)|f(I,\boldsymbol u^0))
$$

其中MI表示互信息，$f(I,u)$ 为对图像$I$ 做参数为$u(a,b)$ 的灰度映射变换，即灰度小于a的置0，大于b的置1，a,b之间的线性变换到0,1之间。后续层应该类似该过程。

# 2019.5.3
---
## 图像中物体识别和位姿估计（刚体且已知3D模型）
### 不同策略
1. 稀疏特征点匹配（描述2D点或点云中的3D点），但是在低纹理场景中使用受限
2. 生成3D模型，在pose空间中采样重投影，在图像中匹配
3. 基于稠密点云的方法：1)ICP；2)用所有可能的点对组合，计算对应的pose并投票，可以做detect或计算pose；3)用所有三点的组合，通过随机森林得到一些候选的pose，再通过优化筛选
4. 局部patch仿射不变描述子，得到相邻patch的邻接关系图模型，用于匹配和重建。（这个忘了在哪看的了，找不到对应论文）

还可以分成整体和局部匹配两种策略，也可以同时使用。全局的匹配可能更稳定，很少出现一对多的情况，但是有遮挡情况表现不好。局部特征匹配则对遮挡处理的好，但是会有多意解。

### ICP
<https://wenku.baidu.com/view/8e8fbf355a8102d276a22fea.html>  
两个三维点集的匹配，迭代的过程：  
1. 求两个点集中的对应点；
2. 求变换关系；
3. 求新的点集位置；
4. 重复上述步骤直到残差小于某阈值

其中对应的点匹配最耗时，可以选择：1.选取所有点；2.均匀采样（Uniform sub-sampling ）；3.随机采样（Random sampling）；4.按特征采样（Feature based Sampling ）；5.法向空间均匀采样（Normal-space sampling）  
匹配方法：1.最近邻点（稳定，速度慢）2.法方向最近邻的点（平滑曲面效果好，对噪音敏感）3.投影法（搜索速度快）

---
**Fast and automatic object pose estimation for range images on the GPU** (2)  
本文即建立3D模型的方法。
1. 用CAD创建物体的3D模型，在位姿空间中采样（在更可能的位姿上更多的采样pose），生成深度图像数据库。用模拟器仿真物体自由下落后出现的位姿，统计直方图，用聚类的方法选择要保存的位姿。  
2. 根据输入的深度图像和数据库估计物体在图像中的大概位置，在该位置上对数据库中的所有位姿计算残差，残差小的作为候选。区域选择方法：深度图中值滤波，提取边缘，计算距离变换，得到可能表示单独物体的像素联通域。根据深度图对每个patch计算平均深度，选大小较大距离较近的几个，用形状中心作为物体坐标系原点，即可得到初始的$T(x,y,z)$。这时候选的patch可能有1-3个。对这几个候选与所有pose计算残差，同时优化Ｔ，得到大概的pose。
3. 用ICP方法优化位姿。

---
**Model Globally, Match Locally: Efficient and Robust 3D Object Recognition** (3)   
本文为稠密点云策略中的，取点对生成候选pose的方法。
1. 全局描述子：用两个点之间的特征和关系构建一个feature F，计算3D模型中所有两点之间的F，按照F聚类，将相似的F对应的点对通过F的hash索引保存在一起。这样通过图像中的点对的F即可找到模型中对应的可能的点对。
2. 在图像中找一个参考点 $s_r$，遍历所有其他点构成点对 $(s_r,s_i)$，计算F，在全局模型中找到相似的点对对应 $(m_r,m_i)$，计算每组对应的transform，在空间 $(m_r,\alpha)$ 中投票。最终得到参考点对应的最可能的几个 $m_r,\alpha$，可据此计算出transform。
3. 由于上述计算前提为s是物体表面上的点，所以要多取几个s，保证有点在表面上。 得到了这么多可能的pose之后，对他们聚类，每类的得分为类内所有pose的得分，即上述投票步骤中所获得的票数。最后得分高的几个类，类内pose平均后输出。

---
**Global Hypothesis Generation for 6D Object Pose Estimation** (3)  
本文即稠密点云策略中，取三点生成pose的方法。  
1. 在一定范围内取三点，通过随机森林判断图像点属于哪个物体及在物体上的坐标，进而计算出pose
2. 不同于常规的pose采样方法（RANSAC），本文用一个全链接的条件随机场判断像素是否是物体上的点，这样可以减小搜索空间
3. 最后用ICP优化。

# 2019.5.10
---
## 三维重建专题
related works  
1. 由带有标注的2D图像恢复3D模型
2. 三维监督学习
3. 多视图重建
4. 多图像训练，单视图深度估计

*总结来说，要从有限信息中恢复三维信息，必须要有先验知识，比如有统计模型或者标注等。*

---
**What Shape Are Dolphins? Building 3D Morphable Models from 2D Images**（1）

从单图像恢复物体的三维模型。注意只能针对一个类型的物体，因为要生成该类物体的3D模型。
1. 从n幅图像中生成3D模型，包括一个平均模型和几个basis变形的模型（由PCA生成）。
2. 来一个新图像时需要提供图像中物体的轮廓，和轮廓上特征点和3D模型上的对应。
3. 优化目标函数为：通过模型模拟生成的轮廓与输入轮廓的匹配误差，控制点的匹配误差，模型平滑和正则化，3D模型上对应于2D轮廓的曲线的连续性；优化参数包括：优化basis模型的组合系数，相机外参，微调模型上的控制点。

文中还介绍了优化的方法，如何找到符合2D图像的轮廓，subdivision surface model等。

---
**Single Image 3D Interpreter Network**(1)

This is a learning based method.Eevery category has a set of basis skeletons which is manually designed. *This method do not need annotations in input image, while the express of object is sparse.* Use a detection net beforehand so we know the category of object in input image.
1. keypoint estimate from input image, and output a heat map of keypoints. With a fine tune approach.
2. 3D Interpreter. find $P,R,T,\{\alpha_k\}$ to fit:

$$X = P(R\sum_{k=1}^K \alpha_kB_k+T)$$

where X is points in image, $B_k$ is basis skeleton which are coordinates of 3D points.

3. project the final skeleton to image and get 2D points.

这个方法类似于pnp，不过特征点提取和匹配过程用网络生成heat map代替。

---
**Reconstructing PASCAL VOC** (1)  
该方法的输入图像需要标注物体的分割，及一些控制点，控制点数量大于上一篇论文的方法，所以重建出来的模型比较稠密。
1. 相机姿态估计。对每个类标注分割和几个控制点，因为物体非刚性，取其中可能不变的部分用SFM生成该类的刚体模型。根据相机投影成像公式，用迭代的方法求相机姿态。
2. 通过轮廓误差细调相机位姿
3. 将所有的相机视角聚类，用PCA选出三个主方向，每个主方向相差15度的实例的轮廓作为该主方向上的轮廓候选之一。在重建时，根据每个主方向上实例的多少，按概率抽选两个主方向中的两个轮廓作为三视图中的两个视图，与输入图像的轮廓组合成三视图。选择多次生成多组三视图。
4. 用三视图重建3D模型，注意这里用的不是同一个实例的三个视图，所有重建会有很大误差。生成多组3D模型，根据模型在三个主方向上的投影，和每个方向上平均投影的相似度进行打分，相似度高的胜。

---
**3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction**（2）  
3D Recurrent Reconstruction  Neural  Network  
用RNN网络记住之前输入的图片，实现单图片到多图片兼容的三维重建，图片越多重建越准确，也有图片多了失败的。输入图片只需要标注bounding box。  
细节：
1. 一张或多张图片输入LSTM，将隐藏层中的特征拿出来经过一个decoder，做3D反卷积，生成occupied confidence map
2. 训练时每个patch用相同数量的图片输入，测试时可以每输入一个图片就输出一个结果。

---
**Learning a Predictable and Generative Vector Representation for Objects**（2）  
先用3D模型训练一个3D模型的encoder-decoder网络，得到3D模型的特征向量F。再训练一个2D图像的encoder，使其生成F。测试时从2D图像经过encoder生成F，再由F经过3D decoder得到3D模型。

---
**Convolutional Mesh Regression for Single-Image Human Shape Reconstruction**  
事先有一个人体的mesh模型，通过CNN提取图片中人体特征，将该特征和mesh模型（一些3D控制点坐标）结合，经过graph CNN，回归出mesh中3D点的坐标，恢复图片中的人体模型。

# 2019.5.15
---
## 恢复三维人体（或者pose）
大部分方法也需要先验模型。机器学习方法中有分步估计（shape and pose），或者end-to-end方法。目前最好的建模方法是SMPL(A skinned multi-person linear model)。  

1. 对图像中的人体关键点计算其3D坐标(pose)  
1.1 提取关键点后估计每个点的深度  
1.2 根据关键点建立basis model，线性组合成模型后投影回图片，最小化残差  
1.3 由2D点之间的信息恢复3D点之间的信息  
2. 重建稠密3D模型(shape)  
2.1 类似上面说的海豚重建，通过统计构建通用模型，然后根据输入的轮廓信息，估计模型参数调整模型
3. Pose & Shape，同时估计pose和shape

### SMPL

$$M(\beta,\theta;\Phi):\mathbb R^{|\theta|\times |\beta|}\Rightarrow\mathbb{R}^{3N}$$  

$\beta$ stands for shape param, $\theta$ stands for pose param.

---
**Ordinal Depth Supervision for 3D Human Pose Estimation**（1.1）  
先经过encoder-decoder结构生成人体关键点和对应的深度，再经过一个网络生成每个关键点的三维坐标。其中深度估计网络的训练用到了两个关键点之间的深度关系，即ordinal depth，具体的点对顺序和选择是固定的，不需要所有点对组合，只取其中的一部分点对组合。

---
**3D Human Pose Estimation from a Single Image via Distance Matrix Regression**（1.3）  
计算图片中关键点两两之间的距离，组成矩阵，经过一个full connect 和一个full conv生成3D关键点两两距离矩阵，根据这个矩阵通过一个半定规划优化方法，解出3D模型。

---
**Learning to Estimate 3D Human Pose and Shape from a Single Color Image**（3）  
1. 图像经过关键点点检测提取关键点，关键点位置经过网络生成pose param  
2. 图像经过分割网络得到轮廓，经过网络生成shape param  
3. pose and shape param组合生成3D模型，重投影回图片，根据点的残差和mask误差，优化网络

---
**End-to-end Recovery of Human Shape and Pose**（3）  
1. 图片经过encoder得到编码，由编码回归出SMPL模型参数，该过程是迭代多次的（因为单次回归不准确）  
2. 回归网络训练的时候可以有3D标注，也可以没有。loss包括3D关键点误差，3D模型误差，关键点重投影误差和模型参数误差。但是注意每个迭代都用ground truth监督的话可能会陷入局部极小值，所以只用了重投影误差和3D误差训练  
3. 因为可能没有3D ground truth，模型可能学习出奇怪的结果，尽管其2D投影是对的。所以最后增加一个对抗先验网络，判断生成的模型是否是真实的人体模型。every joint has a discriminator, 每个分类器有loss，最后总体有loss，同时训练（这里没细看）。

总体loss

$$L = \lambda (L_{reproj}+L_{3D})+L_{adv}$$

---
**Exploiting temporal context for 3D human pose estimation in the wild**  
作者Andrew Zisserman，在同时估计pose & shape的基础上，对多帧图像中的人体做BA，完善了结果，并且用该方法给大量网络视频做了标注，实验表明，其他算法在这些标注上训练后效果变好了。

---
# 2019.5.25
**The graph neural network model**  
GNN 是一种对网络节点或整体做预测的方法。

$$\begin{align}
x_n&=f_w(l_n,l_{co[n]},x_{ne[n]},l_{ne[n]})\\
o_n&=g_w(x_n,l_n)
\end{align}$$

其中 $x_n$ 表示节点状态，$o_n$ 表示节点的特征向量。函数f内参数分别为，节点label，与节点相连的边的label，相邻节点的状态，相邻节点的label。节点的状态估计用迭代的方法计算，在网络形式上表现为递归网络。网络结构和graph的连接方式有关，所以不能处理graph变化的情况。实际上主要是计算节点状态的部分受限于graph结构。类似PointNet的网络虽然也用了GNN，但是它计算point状态不严格依赖points之间的连接关系，实际上点云之间没什么连接关系，或者说关系很弱，关系比较统一。

文章还分析了网络的可导性，计算复杂度。

---
**Training Deep Networks with Synthetic Data: Bridging the Reality Gap by Domain Randomization**  
related works：  
1. 用仿真数据做光流，双目，相机位姿估计等
2. 用真实数据预训练，再在合成数据上训练（固定参数）。文中说这样效果不好。
3. 用真实照片合成数据训练

本文在3的基础上，用Domain Randomization代替真实照片合成，用合成数据预训练，再在真实数据上训练，可以达到更好的效果。DR就是用网络生成足够多的场景情况，其中包含了真实场景情况，所以其训练出的网络可以适应真实的图片。

DR：在被检测物体的模型上随机添加纹理和颜色，选择随机角度，随机背景图片，添加一些随机干扰物体，如多面体（也要添加纹理和颜色），随机数量和位置的点光源，等等。

---
**How do neural networks see depth in single images?**  
这篇文章分析了一个单目深度预测网络的结果和什么因素相关，主要是道路上的图片。包括物体位置，大小，颜色，相机角度等因素。在控制变量的基础上改变这些因素，看深度预测结果有什么变化。

---
**Learning to Count Objects with Few Exemplar Annotations**  
如何在标注不完全的情况下，即图像中很多车，有的车框出来了，有的没框，怎么训练一个准确detect车的网络？因为training data中有较多false negative, so the detector works bad. The authors use other data set which dose not include such category to augment the data set, and train the net for multi stages, auto label positive cases which are not labeled in the origin data set during these stages. 用其他数据集的负样本扩充数据集，在迭代学习中不断标定正样本，实现训练。

---
**Side Window Filtering**  
保边滤波器，不将像素放在滤波器的中心，而是放在边上或角上。对每个待处理的像素，计算八种窗口内的滤波结果，选择和原图差别最小的作为最终结果。八种滤波器位置包括左右，上下，四角。

---
# 2019.5.29
**GIANT PANDA FACE RECOGNITION USING SMALL DATASET**  
在小数据集下实现大熊猫脸识别。用传统的特征提取法。  
1. 特征提取方法：取边缘并和特定的参考边缘对齐，在不同的网格（7种分法）中提取特征
2. 训练分类器：数据集中有N幅图片，n个大熊猫（n<N）。对每个图片，其他所有图片和它对齐并提取特征，所有和它有相同ID的图片标注为1,其他为-1,训练分类器，这样共训练N个。
3. test：待识别目标图片和每个训练集中的图片对齐并N个提取特征，分别经过N个分类器，输出结果最大的图片对应的ID就是结果。

## 人脸识别专题


---
## 语义分割
SegNet：全卷积网络，encoder时记录max pooling的位置给到decoder。base net VGG16。

U-net：将encoder的feature复制并concatenate到decoder中。

ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation。全卷积网络，修改残差模块，将 $3\times 3$ 卷积变成 $1\times3$ 卷积加 $3\times 1$ 卷积，减少参数，加快计算。

PSPNet：图像经过CNN生成feature，pooling成不同尺寸再upsampling，concatenate，convolution。

refineNet：在decoder时相邻几个层级的feature做融合，高级feature upsampling，与低级feature sum，融合后的feature经过级联式的pooling得到不同scale的pooling结果，然后相加（维度不同怎么加的，文中没说），最后经过三个res model。

deeplab：空洞卷积获取高级feature的同时避免尺寸过小，使网络更深。最后增加条件随机场（CRF）平滑分割结果。

---
## 实例分割
Mask R-CNN：用检测实现实例分割，关键步骤为RoIAlign layer，提取roi feature时不做量化，用双线性差值。

Recurrent Instance Segmentation：用LSTM生成feature，用注意力机制限制分割区域，使得每次分割一个实例。递归地分割出所有实例。

Instance Segmentation of Indoor Scenes Using a Coverage Loss：先分割，再用树的方法融合像素，分割实例。开始时每个像素自己一组作为graph中的节点，相邻像素之间的边表示他们属于两个区域的概率。通过不同的阈值融合相邻像素得到由粗到细的分割结果。

Pixel-level Encoding and Depth Layering for Instance-level Semantic Labeling：通过预测每个像素对所属物体可见部分中心的方向，来确定像素的归属。

# 2019.5.31
**P3SGD: Patient Privacy Preserving SGD for Regularizing Deep CNNs in Pathological Image Classification**  
关于网络的正则化（提高泛化能力）：
1. 在输入端的正则化：data augmentation
2. 模型中的正则化：参数正则化，drop out，卷积feature dropout
3. 在输出位置正则化：训练时随机修改label
4. 本文是在更新参数时增加随机性。

本文同时使用了private SGD保护数据来源的隐私信息。用有上限的函数序列拟合目标函数，并在其中添加噪声。本文中好像是将求得的梯度除以一个上限，然后添加高斯噪声，再更新参数。
