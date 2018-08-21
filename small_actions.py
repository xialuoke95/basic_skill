#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 13:40:35 2018

@author: zy
@content: small_actions
"""

## 同样元素的list生成
a = [1] * 4
b = [7,8]
c = [7,8,9,6]
## zip的时候，如果一方不够长，另一方会自动缩短
zip(a,b)  
zip(a,c)

a = [1,2,3]
b = [4,5,6]
for aa,bb in zip(a,b):
    aa += bb    
a

a = [[1,2],[3,4]]
b = [[5,6]]
for aa,bb in zip(a,b):
    aa += bb
a
