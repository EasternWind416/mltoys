# Random forest

#### 算法逻辑

* subsample(dataset, sample_size);

  进行有放回的下采样，将数据进行重新组合。sample_size表示重新组合之后的数据集合大小。
  
* build_tree(dataset, max_depth, min_size, n_features, depth);

  分别进行建树。max_depth表示树的最大深度，min_size表示最后剩余多少数据应该直接作为叶子，n_features表示选择多少个特征进行分割。

  * 得到最优分割的特征和数据：

  	1. 从所有特征中筛选n_features个特征作为对比特征让那个（可以重复）

  	2. ```python
     对于每个特征：
     	对于数据的每行（即每个value）：
         	使用test_split(dataset, feature, value)进行分割；
             计算gini系数，比较最小的gini系数，作为最优的分割；
        ```
    
  * 进行分割、建树：

    1. 分割后的左子树或者右子树是空的，则直接按照最多的标签生成叶子结点；
    2. 如果达到最大深度，则直接按照最多的标签生成叶子结点；
    3. 对于左子树or右子树：
       * 若数据少于min_size，则生成叶子结点
       * 否则，继续迭代。

    