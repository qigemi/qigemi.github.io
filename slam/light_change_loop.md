---
layout: post
---

#光照变化下的闭环
related works：  
1. 用激光雷达或深度相机补充信息
2. 用HDR图像得到不同曝光下的信息（不能handle快速运动）
3. 用光照鲁棒描述子
4. 时实调整相机曝光
5. 根据物体的自相似性查找场景
6. 图像序列匹配（而不是一对多匹配）
7. 图像高级语义信息用于匹配
8. 用CRF整体匹配特征点而不是一对一匹配
9. 图像灰度仿射变换

主要大概两种思路：1.估计出亮度仿射变换系数；2.训练一种光照鲁棒的描述子或者词袋。

---
**FAB-MAP: Probabilistic Localization and Mapping in the Space of Appearance**  
http://www.robots.ox.ac.uk/~mjc/Papers/IJRR_2008_FabMap.pdf

Chow and Liu prove that the maximum-weight spanning tree (Cormen et al. 2001) of the mutual information graph will have the same structure as the closest tree-structured Bayesian network  to approximate discrete distributions.  
用surf训练一个词袋，然后用word建一个chow-liu树，节点是word，边表示两个word之间共视的图像个数，然后按照个数从大到小生成树。输入新图时用该树计算不同location的概率，即有相似特征点排布的场景概率更高。这篇没说对光照鲁棒。

---
**Robust Visual Localization in Changing Lighting Conditions**(4)  
http://brian.coltin.org/pub/kim2017robust.pdf  
通过BRISK特征估计亮度，然后调整相机曝光。

---
**Learning Place-and-Time-Dependent Binary Descriptors for Long-Term Visual Localization**  
通过标注相同地点不同光照下的图像，训练二值描述子的采集方法，类似于下面的 fine vocabulary 方法，只不过FV训练的是描述子分类器，这篇是训练生成描述子。

---
**SeqSLAM: Visual Route-Based Navigation for Sunny Summer Days and Stormy Winter Nights**(6)  
inspired by RatSLAM: 在序列中判断可能的loop，而不是找到最像的一个，避免了严格的特征匹配。该过程只需要1000个像素，且不需要提取特征，只计算一个和灰度相关的描述即可。该方法不能适应白天和夜晚的光照变化。使用序列同样也可以结合特征（如SIFT）。  
1. 差分图像D做归一化，可以增加匹配概率
2. 假设每次经过相同场景时速度相近。同样需要保存landmark，将整幅图像压缩，每隔相同时间间隔依次保存。loop时取最近时间内的n幅图像，与所有lm做差，得到一个矩阵，横轴是最近的n幅图像，纵轴是所有lm，如果能loop上，则会出现一个直线型的峡谷（即残差最小）。找到这种峡谷路径，通过残差求和找到显著小的路径就可以认为loop了。

该方法大概需要以下几点前提：图像不能旋转，每次经过相同场景速度相近，视角相近，比较局限在公路上。还有保存量较大，地图规模大时可能计算量也很大。

---
**Matching Local Self-Similarities across Images and Videos**(5)  
http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_CVPR_2007/data/papers/0230.pdf  
这篇主要讲类似模板匹配，给一个形状，从图片或视频中得到热图，找出位置。输入F，在G中查找。在F和G中在不同尺度提取局部形状描述子，去掉G中无纹理或重复纹理区域，组成全局特征，在其中查找和F相似的特征组成，生成热图。  
其中的描述子是基于patch的而不是像素。在像素p周围选8\*8个patch，每个patch可能5\*5，每个patch和中间的计算sum of square differences，得到残差，然后按照极坐标(20个角度，4个半径)排列这些残差，组成描述子。

---
**Towards illumination invariance for visual localization**(3)  
http://www.ananth.in/Home_files/Ranganathan13icra.pdf  
使用了 [Learning a Fine Vocabulary](http://cmp.felk.cvut.cz/~chum/papers/mikulik_eccv10.pdf) 的方法。主要解决光照变化下，描述子变化大的情况下如何匹配对应点。  
在不同光照下采集图片，手动确定那些描述子是同一个3D点在不同光照下的结果，计算词袋中检测到 word A 实际是 word B 的概率。 **本质上是有监督地训练了一个分类器！** 可能有限的训练效果也有限，看迁移能力了。

---
**VISUAL LOCALIZATION USING SPARSE SEMANTIC 3D MAP**(7)  
给每个3D点分配一个语义，在PnP时通过2D-3D语义匹配加权做RANSAC，即在常规匹配下加入语义匹配权重。该文章在CVPR定位比赛中handle了不同季节和天气的定位问题，拿到第一名。

---
**Robust Visual Odometry to Irregular Illumination Changes with RGB-D camera**(9)  
根据深度信息得知平面，在平面上选patch并认为光照一样，通过和key frame的光度比较分别调整每个patch的仿射系数。

**stereo LSD**(9)  
和key frame优化位姿的时候就增加灰度仿射系数a，b，（ $I'=I*a+b$ ），分别优化ab和位姿。用不同的误差容忍度，ab的容忍度低，对于误差大的数据需要剔除；优化位姿时误差大的数据只降低其权重，不剔除。

闭环时在定位附近根据累积误差划定一个范围，范围内所有候选的 key frame 做上述优化，因为估计深度，所以可以求出 $T_{ij}, T_{ji}$ 两个位姿，两个位姿相差小的认为loop。其中在优化时，从最顶层金字塔开始匹配和优化，只有该层匹配效果好才进行下一层，可以提高速度。

[blog](https://blog.csdn.net/j10527/article/details/69538707)

---
**Dealing with Shadows: Capturing Intrinsic Scene Appearance for Image-based Outdoor Localisation**  
本文是去掉影子的方法，通过色度和色温将彩色图像投影到一个一维空间（亮度不变色彩空间），产生灰度图像，阴影变化只会在该投影方向上变化，所以不会影响产生的灰度图。但是不知道对灰度图怎么应用。

---
**[Real-Time SLAM Relocalisation](http://www.robots.ox.ac.uk/ActiveVision/Publications/williams_etal_iccv2007/williams_etal_iccv2007.pdf)**  
用整幅图做loop。高斯模糊后剪裁成80*60,用SSD找最小的匹配。特征点匹配是基于词袋的，但是词袋的距离不是汉明距离，而是训练得到分类器。首先选N种描述子比较配对，对应N棵决策树，每棵树根据该描述子找到一个叶子，叶子是每个词的概率，然后N棵树结果相加，得到最终的词分类。训练过程就是手动标注一个数据集，其中有同一个词在不同光照和视角下的N种描述子，通过每棵树找到叶子后，在该概率分布上找到自己的词，加一。优雅啊。

**[Vision-Based Global Localization and Mapping for Mobile Robots](http://www.ent.mrt.ac.lk/iml/paperbase/TRO%20Collection/TRO/2005/june/8.pdf)**  
在平面运动假设下loop，似乎只是用map点投影匹配的。

---
about me
