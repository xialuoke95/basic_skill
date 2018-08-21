# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

pd.DataFrame([[1,2,3],[4,5,6]]) # 按行输入
pd.DataFrame([1,2,3],[4,5,6]) # list1 = col，list2 = index

pd.DataFrame({'tag':[0,1,2,3],'pred':[4,5,6,7]}) # 按列输入

pred_result = pred_result.sort_values(by = 'bad_prob', ascending=False)
# ascending = False: 降序输出，True：升序输出
# by = col_name: 排序列
# 多列排序
pd.read_csv('', header = None, index_col = None)
df.to_csv('', index = False)

aa = pd.read_csv('20180101_20180730_report.csv')
bb,cc = train_test_split(aa.index)

aa.columns
type(aa['a'])
np.array(aa['a'])
np.array(aa[['a','b']])
aa.shape[0]
aa.shape
len(aa)
aa.iloc[:,0:1]

import numpy as np
np.array([[1,2],[4,5]])
np.array([[1,2],[4,5]])[0]


