#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:17:21 2018

@author: zy
"""

### 危险：find_near_index;tolist;  delta_lis, check

## 这些能练到一小时写出来勉强算是出师。
## 其实不用dataframe结构也完全可行！
import pandas as pd
import numpy as np
import copy
import time
start = time.clock()
c = lis_diff(a,b) # 4.2 s!!
elapsed = (time.clock() - start)

start = time.clock()
try_data = get_impr_df(data,6548551597322928648)
try_metric = GidImprMetric()
try_metric.run(try_data)
elapsed = (time.clock() - start)
## 也不过是0.01s，还可以接受
# lis_diff怎么会要这么久

# from get_info_from_data import GidImprDf,get_impr_df
NON_SENSE_VALUE = -1
MIN_TO_SECOND = 60
Start_Index = 0
End_Index = -1

pdShapeRowIndex = 0
pdShapeColumnIndex= 1
getInDayTimeModInt = 10000
npWhereIndexListIndex = 0
pdIterrowsInfoIndex = 1

# metric_name与metric：将name调用与值调用区分开来
basicMetrics = ["read_impr_ratio","read_impr_ratio_lis","impr_density_lis"]
basicInfoKeys = ["group_id","total_impr","total_read","first_impr_time","last_impr_time"]
# lisInfoKeys = ["impr_lis","read_lis","impr_in_interval_lis","read_in_interval_lis","imprtime_lis"]
needColumns = ['group_id','impr_time','impr_time_in_view','impr_count','read_count','impr_accumu','read_accumu']
default_minutes_from_first_impr = 1440
default_minutes_interval = 30

data = pd.read_csv("~/toutiao_not_online/train.csv")
data = pd.read_csv("~/toutiao_not_online/vulgar_title_offline_analysis/train_real.csv",header = None)

data.columns = needColumns

len(list(set(data["group_id"].tolist())))
len(data["group_id"].unique())
count_gid = data.groupby("group_id").count()
gid = pd.read_csv("~/toutiao_not_online/vulgar_title_offline_analysis/gid_6516684744900477448.csv")
len(list(set(gid["gid"].tolist())))

a = list(set(data["group_id"].tolist()))
b = list(set(gid["gid"].tolist()))


len(lis_diff(a,b))
#237
len(lis_diff(b,a))
#2
多出来的237？
# 29200

np.where(["group_id"] == c[0])[npWhereIndexListIndex]

# python的for line in lines可以不是迭代器而是直接能识别出是list？？
import csv  
with open('test.csv','rb') as myFile:  
    lines=csv.reader(myFile)  
    for line in lines:  
        print line
# writer.writerow会事先清空文本，如果要append，再说
# 有line_num属性
# 但我记得·不用这么麻烦: 用next取一下就返回好东西、直接进入下一个东西
# 没有keys可能是因为缺乏列名
        

## 目前只针对整数

# 如何把key加成私有的？
# 如何在get初始化的时候允许变动间隔时间？
# 拆多了以后参数传递一直带着的问题？
# 针对接口而非实现编程，把接口拆的越散越好

## lis的长度不完整也有可能是中间漏了，不一定是最后结尾部分漏了！

def gen_time_lis(start_time,time_interval,time_lis_len):
    time_lis = []
    for num in range(time_lis_len):
        time_lis.append(start_time + num * time_interval)
    return time_lis

# 假设lis2是lis1的子集
# remove是一个直接对对象操作的命令，使用了会改变对象，所以此处应该复制一下！
    # 简单的用等于号只是传了引用，仍然会在原对象上操作
# 为什么这么慢
# remove 仅仅 删除一个值的首次出现。 

# 如果在 list 中没有找到值，程序会抛出一个异常

# 最后，你遍历自己时候对自己的内容进行删除操作，效率显然不高，还容易出现各种难debug的问题
def lis_diff(lis1,lis2):
    lis = copy.deepcopy(lis1)
    for val in lis2:
        try:
            lis.remove(val)
        except:
            pass
    return lis

def lis_diff_2(lis1,lis2)
    

def pad_vec_with_one_val(vec, non_sense_val, val):
    for i in range(len(vec)):
        if (vec[i] == non_sense_val): vec[i] = val
    return vec
        
def pad_vec_with_near_val(vec, non_sense_val):
    for i in range(1,len(vec)):
        if (vec[i] == non_sense_val): vec[i] = vec[i - 1]
    return vec

## 这种pad没有整体的pad好；统计一下pad了多长
## 4.27就结尾了，所以出现这种操蛋情况

def gen_name_vec(title,length):
    name_vec = []
    for i in range(length):
        name_vec.append(title + "_" + str(i))
    return name_vec

def gen_metric_vec_len_dict(metric_vec_len_lis):
    metric_vec_len_dict = {}
    length = len(metric_vec_len_lis)
    if length != len(basicMetrics):
        raise Exception("length of provided metric_vec_len_lis not pipei")
    else:
        for i in range(length):
            metric_vec_len_dict.update({basicMetrics[i]:metric_vec_len_lis[i]})
    return metric_vec_len_dict
        
# vec += str的时候会把str拆开变成字节!此时必须用append 这时候的+=没有把str视为一个字符串而是视为了一个list  
# += 本来就是vec与vec之间的行为，而append是vec与元素之间的行为      
def gen_impr_metric_vec_name(metric_vec_len_dict):       
    name_vec = []       
    for metric_name in basicMetrics:
        name_vec += gen_name_vec(metric_name, metric_vec_len_dict.get(metric_name))
    return name_vec
# return与方法内修改的关系，经常写错
# 覆写关系：在参数层使用的尽量不要在循环变量或者函数体内使用
class GidImprMetric(object):
    def __init__(self):
        metrics = {}
        for metric_name in basicMetrics:
            metrics[metric_name] = None
        self.__dict__.update(metrics)
        self.__dict__.update({'need_info':{}})

    def __setattr__(self,key,value):
        if hasattr(self,key):
            self.__dict__[key] = value
        else:
            raise Exception("metric not exist") 
    
    # 此处可以根据需要自由修改
    def get_need_info(self,gid_impr_df):
        GidImpr = GidImprDf(gid_impr_df) ## 自动执行gid_impr_df格式检查
        info = {}
        info.update({'basic_info':GidImpr.get_basic_info()})
        info.update({'impr_2h_10min_lis':GidImpr.get_impr_in_interval_lis(120,10)})
        info.update({'read_2h_10min_lis':GidImpr.get_read_in_interval_lis(120,10)})
        info.update({'impr_day_30min_lis':GidImpr.get_impr_in_interval_lis(1440,30)})

        self.need_info.update(info)
    

# 目前，由数据得到的lis最多缺兵少将，不至于多出什么    
    
    def get_time_lis_should_be(self,minutes_from_first_impr, minutes_interval):
        time_interval = minutes_interval * MIN_TO_SECOND
        time_lis_should_be_len = minutes_from_first_impr / minutes_interval
        time_lis_should_be = gen_time_lis(self.need_info.get("basic_info").get("first_impr_time"),\
                     time_interval, time_lis_should_be_len)
        return time_lis_should_be
    
    # def compute_time_lis_len(self,minutes_from_first_impr):
    
    def init_metric(self, metric_name, init_time_lis, init_metric_lis, minutes_from_first_impr, minutes_interval):
        time_lis_should_be = self.get_time_lis_should_be(minutes_from_first_impr, minutes_interval)
        init_val = [NON_SENSE_VALUE] * len(time_lis_should_be)       
        for time,metric in zip(init_time_lis,init_metric_lis):
            init_val[time_lis_should_be.index(time)] = metric
        self.__dict__.update({metric_name:init_val})
    
    def compute_read_impr_ratio(self):
        total_read = self.need_info.get("basic_info").get("total_read")
        total_impr = self.need_info.get("basic_info").get("total_impr")
        self.read_impr_ratio = [float(total_read)/total_impr]
        
    def compute_read_impr_ratio_lis(self,minutes_from_first_impr = 120, minutes_interval = 10, pad_vec = True):
        read_lis = self.need_info.get("read_2h_10min_lis").get("lis")
        impr_lis = self.need_info.get("impr_2h_10min_lis").get("lis")
        time_lis = self.need_info.get("read_2h_10min_lis").get("time")
        read_impr_ratio_lis = [float(read)/impr for read,impr in zip(read_lis,impr_lis)]

        self.init_metric("read_impr_ratio_lis",time_lis,read_impr_ratio_lis,\
                                                    minutes_from_first_impr,minutes_interval)
        ## 为保证vec等长
        ## 注意：指标是按照lis形式存储的！！！
        if pad_vec:
            insert_val = self.read_impr_ratio[Start_Index]
            self.read_impr_ratio_lis = pad_vec_with_one_val(self.read_impr_ratio_lis, NON_SENSE_VALUE, insert_val)
        
           
    def compute_impr_density_lis(self,minutes_from_first_impr = 1440, minutes_interval = 30, pad_vec = True):
        def impr_density(impr,time_interval,total_impr,total_time):
            return (float(impr)/total_impr) / (float(time_interval)/total_time)
        impr_lis = self.need_info.get("impr_day_30min_lis").get("lis")
        time_lis = self.need_info.get("impr_day_30min_lis").get("time")
        time_interval = minutes_interval * MIN_TO_SECOND
        total_impr = sum(impr_lis)
        total_time = time_lis[End_Index] - time_lis[Start_Index]        
        impr_density_lis = [impr_density(impr,time_interval,total_impr,total_time) for impr in impr_lis]
        
        self.init_metric("impr_density_lis",time_lis,impr_density_lis,\
                                                    minutes_from_first_impr,minutes_interval)
        if pad_vec:
            self.impr_density_lis = pad_vec_with_near_val(self.impr_density_lis, NON_SENSE_VALUE)

## 只差一个定顺序的24h指标了    
    
    def run(self,gid_impr_df):
        self.get_need_info(gid_impr_df)
        for metric_name in basicMetrics:
            eval('self.compute_' + metric_name + '()')
        
    def get_metrics(self):
        metric_vec = []
        for metric_name in basicMetrics:
            metric_vec += self.__dict__.get(metric_name)
        return metric_vec
        
        
#########3 得到数据；继承metrics类；在统一的地方管理所用指标；用不同类修改指标的不同部分？ ############
# 名字起的好，拆的细是正向效果，反之则是反向效果
        
        
fail: 6518217509370331656
fail: 6546446971459076615
fail: 6531991489331855880
fail: 6543867788019106318
fail: 6515311016320958979
fail: 6516744274963333646
fail: 6530957202134401543
fail: 6540540730245382669
fail: 6517038573655441933
fail: 6520932041553347080
fail: 6545673353879880195
fail: 6535756959293899277
fail: 6547631203346809348
fail: 6517210424474075655
fail: 6541951271882981895
fail: 6520492746250125832
join数据花了：
1210.016753
len_wrong: 0
fail: 16
        
        
