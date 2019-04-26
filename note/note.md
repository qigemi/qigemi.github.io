---
layout: post
---
## 2019.4.25
### PointNet:
对三维点云的分类和分割网络，文章讨论了输入点顺序改变时网络性能的不变性，没看懂。<br/>
1. 输入$n\times3$矩阵，n为点数，3为x,y,z坐标，经过一个变换矩阵（由T-net生成）将点云变换到相对标准的位置，避免整体transform的影响。<br/>
2. 变换后经过多层感知机（MLP）分别对每个点进行处理，生成$n\times64$ feature，再对feature做（T-net）transform。<br/>
3. 对上述feature再做MLP，max pooling，FC等操作得到全局特征向量，由此可做分类。<br/>
4. 将该向量和$n\times64$ feature concatenate,经过MLP得到每个点的分类。<br/>

### FoldNet：
基于pointnet的改进，用encoder-decoder结构做无监督学习，可以生成一个点云的特征向量，还可以通过该向量将一个点云变换为另一个。<br/>
其中对pooling方法也有改进，他不是全局pooling，而是利用最近邻图（NNG）对某点附近的几个点做pooling（max or average）。

### On-the-Fly Adaptation of Regression Forests for Online Camera Relocalisation  
一种在线训练回归森林做相机重定位的方法。  
related works：  
1. 图像匹配的方法，对图像生成描述子（词袋或其他向量描述子）  
2. key points匹配，生成的3D points和当前图像的2D points 匹配（RANSAC，Kabsch等）  
3. 其他方法，如ORB-SLAM结合上述两种方法，用神经网络直接回归6自由度pose，在当前位置附近生成3D points，将其和全局3D points匹配。  
4. 回归森林方法，和词袋类似，需要预先训练一棵回归树，应用时用来做3D-2D匹配。

本文的改进是在普遍场景中训练回归森林，然后去掉叶子，应用时online训练叶子。
