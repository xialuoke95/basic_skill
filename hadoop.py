# -*- coding: utf-8 -*-

hadoop fs -mkdir /user/zhangyu.95/debug/online

# put path : new file with default, previous name
# put path + name: new file with given name
hadoop fs -put model_article_0701_0703_v2_01.txt /user/zhangyu.95/debug/online/online_model.param
