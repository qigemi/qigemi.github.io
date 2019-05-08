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
