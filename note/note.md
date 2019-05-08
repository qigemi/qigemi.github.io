---
layout: post
---
## 2019.4.25
-------
**PointNet**  
对三维点云的分类和分割网络，文章讨论了输入点顺序改变时网络性能的不变性，没看懂。<br/>
1. 输入$n\times3$矩阵，n为点数，3为x,y,z坐标，经过一个变换矩阵（由T-net生成）将点云变换到相对标准的位置，避免整体transform的影响。<br/>
2. 变换后经过多层感知机（MLP）分别对每个点进行处理，生成$n\times64$ feature，再对feature做（T-net）transform。<br/>
3. 对上述feature再做MLP，max pooling，FC等操作得到全局特征向量，由此可做分类。<br/>
4. 将该向量和$n\times64$ feature concatenate,经过MLP得到每个点的分类。<br/>
---
**FoldNet**  
基于pointnet的改进，用encoder-decoder结构做无监督学习，可以生成一个点云的特征向量，还可以通过该向量将一个点云变换为另一个。<br/>
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

## 2019.4.29
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

\[
\boldsymbol{u}^* = \mathop{\arg\min}_{\boldsymbol u} MI(I^*, f(I,\boldsymbol u)|f(I,\boldsymbol u^0))
\]
其中MI表示互信息，$f(I,u)$ 为对图像$I$ 做参数为$u(a,b)$ 的灰度映射变换，即灰度小于a的置0，大于b的置1，a,b之间的线性变换到0,1之间。后续层应该类似该过程。

##2019.5.3
---
###图像中物体识别和位姿估计（已知3D模型）
####不同策略
1. 稀疏特征点匹配（描述2D点或点云中的3D点），但是在低纹理场景中使用受限
2. 生成3D模型，在pose空间中采样重投影，在图像中匹配
3. 基于稠密点云的方法：1)ICP；2)用所有可能的点对组合，计算对应的pose并投票，可以做detect或计算pose；3)用所有三点的组合，通过随机森林得到一些候选的pose，再通过优化筛选
4. 局部patch仿射不变描述子，得到相邻patch的邻接关系图模型，用于匹配和重建。（这个忘了在哪看的了，找不到对应论文）

还可以分成整体和局部匹配两种策略，也可以同时使用。全局的匹配可能更稳定，很少出现一对多的情况，但是有遮挡情况表现不好。局部特征匹配则对遮挡处理的好，但是会有多意解。

####ICP


---
**Fast and automatic object pose estimation for range images on the GPU**  
本文即建立3D模型的方法。
1. 用CAD创建物体的3D模型，在位姿空间中采样（在更可能的位姿上更多的采样pose），生成深度图像数据库。用模拟器仿真物体自由下落后出现的位姿，统计直方图，用聚类的方法选择要保存的位姿。  
2. 根据输入的深度图像和数据库估计物体在图像中的大概位置，在该位置上对数据库中的所有位姿计算残差，残差小的作为候选。区域选择方法：深度图中值滤波，提取边缘，计算距离变换，得到可能表示单独物体的像素联通域。根据深度图对每个patch计算平均深度，选大小较大距离较近的几个，用形状中心作为物体坐标系原点，即可得到初始的$T(x,y,z)$。这时候选的patch可能有1-3个。对这几个候选与所有pose计算残差，同时优化Ｔ，得到大概的pose。
3. 用ICP方法优化位姿。
---
**Model Globally, Match Locally: Efficient and Robust 3D Object Recognition**   
本文为稠密点云策略中的，取点对生成候选pose的方法。
1. 全局描述子：用两个点之间的特征和关系构建一个feature F，计算3D模型中所有两点之间的F，按照F聚类，将相似的F对应的点对通过F的hash索引保存在一起。这样通过图像中的点对的F即可找到模型中对应的可能的点对。
2. 在图像中找一个参考点 $s_r$，遍历所有其他点构成点对 $(s_r,s_i)$，计算F，在全局模型中找到相似的点对对应 $(m_r,m_i)$，计算每组对应的transform，在空间 $(m_r,\alpha)$ 中投票。最终得到参考点对应的最可能的几个 $m_r,\alpha$，可据此计算出transform。
3. 由于上述计算前提为s是物体表面上的点，所以要多取几个s，保证有点在表面上。 得到了这么多可能的pose之后，对他们聚类，每类的得分为类内所有pose的得分，即上述投票步骤中所获得的票数。最后得分高的几个类，类内pose平均后输出。

---
**Global Hypothesis Generation for 6D Object Pose Estimation**  
本文即稠密点云策略中，取三点生成pose的方法。  
1. 在一定范围内取三点，通过随机森林判断图像点属于哪个物体及在物体上的坐标，进而计算出pose
2. 不同于常规的pose采样方法（RANSAC），本文用一个全链接的条件随机场判断像素是否是物体上的点，这样可以减小搜索空间
3. 最后用ICP优化。
