## SVM-SMO-Algorithm

1. 寻找一对alpha

2. $f(x_i)=wx_i+b=\sum_k\alpha_ky_kx_kx_i+b$

    $e_i=f(x_i)-y_i$

3. 当点不满足条件时候：

   * **点在边界之间** 且 $\alpha<C$

   * **点在边界两侧** 且 $\alpha > 0$

4. $f(x_j)=wx_j+b=\sum_k\alpha_ky_kx_kx_j+b$

    $e_ji=f(x_j)-y_j$

5. 选定范围L和H：
   * i，j异侧的时候：$L=\max(0, \alpha_j-\alpha_i),\ H=\min(\alpha_j - \alpha_i +C, C)$
   * i，j同侧的时候：$L=\max(0,\alpha_j+\alpha_i-C),\ H=\min(\alpha_j+\alpha_i,C)$

6. $\eta=K_{ii}+K_{jj}-2K_{ij}>0$
7. $\hat{\alpha_j}+=y_j(e_i-e_j)/\eta,\\ \hat{\alpha_i}+=y_iy_j(\alpha_j-\hat{\alpha_j})$

8. 确定b：

   $b_1=b-e_i-y_i(\hat{\alpha_i}-\alpha_i)K_{ii}-y_j(\hat{\alpha_j}-\alpha_j)K_{ij}\\ b2=b-e_j-y_i(\hat{\alpha_i}-\alpha_i)K_{ij}-y_j(\hat{\alpha_j}-\alpha_j)K_{jj}$

   * 当$0<\alpha_i<C$:

     $b=b_1$
     
   * 当$0<\alpha_j<C$:
   
     $b=b_2$
     
   * 两者都在界内：
   
     $b=b_1=b_2$
     
   * 两者都在界外：
   
     $b={b_1+b_2 \over 2}$
   
     