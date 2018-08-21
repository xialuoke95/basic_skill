1#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 20:56:34 2018

@author: zy
"""
# 关于dataframe的更好写法？
# 把judgement单独抽象一个类？

NON_SENSE_VALUE = -1
pdShapeRowIndex = 0
pdShapeColumnIndex= 1
getDayTimeModInt = 10000
imprNearPercent = 0.05
ratioNearPercent = 0.1
npWhereIndexListIndex = 0
pdIterrowsInfoIndex = 1
startIndex = 0

# 重新定义关键字的方式，比如定义True为T？


import numpy as np
import pandas as pd
import dateutil
from datetime import datetime


vulgar_impr_hour = pd.read_csv("~/toutiao_not_online/vulgar_impr_hour.csv")
vulgar_impr_hour.columns = ['group_id','impr_time','impr_accumu','read_accumu']

title_impr_hour = pd.read_csv("~/toutiao_not_online/title_impr_hour.csv")
title_impr_hour.columns = ['group_id','impr_time','impr_accumu','read_accumu']
# title_impr_hour.to_csv("~/toutiao_not_online/title_impr_hour.csv",index = False)

high_impr_hour = pd.read_csv("~/toutiao_not_online/high_impr_hour.csv")
## csv > 数据框 > R

# 注意下载utf-8的东西
high_impr_info = pd.read_csv("~/toutiao_not_online/high_impr_0420.csv")

## 如何更符合逻辑地按行创建数据并转化为pandas？
## 传引用？
## 上一版gid_begin的写法，由于到最后一行时还等于gid_begin,而且没有下一行的机会来发现最后一行也是有意义的行，故而会漏掉一个gid及对应impr
# 比较依赖dataframe数据结构，但有一点感觉了

    
    
def get_user_action_trend_vecs(impr_df,gid):
    gid_impr_df = get_gid_impr_df(impr_df,gid)
    GidImprInfo(gid_impr_df).getBasicInfo()
    GidImprInfo()
    
    

def get_nrow(impr_df):
    return impr_df.shape[pdShapeRowIndex]

def get_impr_total(impr_df):
    
    nrow = get_nrow(impr_df)
    gid_lis = []
    impr_time_lis = []
    impr_lis = []
    read_lis = []
    
    for row_index in range(nrow):
        gid = impr_df.iloc[row_index,]["group_id"]
        gid_next = impr_df.iloc[row_index + 1,]["group_id"] if (row_index < nrow - 1) else NON_SENSE_VALUE
        if gid != gid_next:
            gid_lis.append(impr_df.iloc[row_index,]["group_id"])
            impr_time_lis.append(impr_df.iloc[row_index,]["impr_time"])
            impr_lis.append(impr_df.iloc[row_index,]["impr_accumu"])
            read_lis.append(impr_df.iloc[row_index,]["read_accumu"])

    data = {'gid':gid_lis,'impr_time':impr_time_lis,'impr':impr_lis,'read':read_lis}
    frame = pd.DataFrame(data)
    return frame

def get_gids_lis(impr_total):
    return impr_total["gid"].tolist()

def get_gid_impr_df(impr_df,gid):
    return impr_df.iloc[np.where(impr_df["group_id"] == gid)[npWhereIndexListIndex],]


    
## 要加三个read_impr_ratio，是因为哪里的接口写死了？如何改进?
    # df中的列，成员，比较对象三者合一;
    # judge的取名也可以看出来，每条judgement需要重新来，这不对
# 如何定义一组类似函数？ 函数族？
# judgement类？
# 构造函数不能重载，当面对想传入两种不同参数来构造的时候怎么解决？
# 通过本来就定义不同的类型，只要具有相应方法就可以调用的方式重载？
# 写一个装饰器？晚上看看
# 依赖impr_df 结构
# a,b的比较物取名方式？
class GidImprInfo:
    def __init__(self,group_id,start_impr_time,end_impr_time,impr_total,read_impr_ratio):
        self.gid = group_id
        self.start_impr_time = start_impr_time
        self.end_impr_time = end_impr_time
        self.impr_total = impr_total
        self.read_impr_ratio = read_impr_ratio

class GidImprDf:
    def __init__(self,impr_df):
        self.impr_df = impr_df
        
    # def __repr__(self):  表示打印；和str的区别？
    def __call__(self):
        start_row_index = 0
        end_row_index = get_nrow(self.impr_df) - 1
        group_id = self.impr_df.iloc[start_row_index,]["group_id"]
        start_impr_time = self.impr_df.iloc[start_row_index,]["impr_time"]
        end_impr_time = self.impr_df.iloc[end_row_index,]["impr_time"]
        impr_total = self.impr_df.iloc[end_row_index,]["impr_accumu"]
        read_impr_ratio = self.impr_df.iloc[end_row_index,]["read_impr_ratio"]
        return group_id,start_impr_time,end_impr_time,impr_total
    # 此时返回的相当于一个元组
        

def is_time_near(compare_time_a,compare_time_b,near_time_interval = 30):
    return True if abs(compare_time_a % getDayTimeModInt - compare_time_b % getDayTimeModInt) <= near_time_interval else False

# 初始化的时候是字典，字典无序！还没得到初始化值
# def is_impr_near(compare_impr_a,compare_impr_b,near_impr_interval = 0.05 * compare_impr_a):
def is_impr_near(compare_impr_a,compare_impr_b,near_impr_interval = 10000,near_impr_interval_related_impr_a = True):
    near_impr_interval = imprNearPercent * compare_impr_a if near_impr_interval_related_impr_a else near_impr_interval  
    return True if abs(compare_impr_a - compare_impr_b) <= near_impr_interval else False

def is_read_impr_ratio_near(compare_ratio_a,compare_ratio_b,near_ratio_interval = 0.01,near_ratio_interval_related_ratio_a = False):
    near_ratio_interval = ratioNearPercent * compare_ratio_a if near_ratio_interval_related_ratio_a else near_ratio_interval  
    return True if abs(compare_ratio_a - compare_ratio_b) <= near_ratio_interval else False

## 像这样的函数，就既可以存活在过程中，也可以封装到类中去
def is_similar_gid(gid_impr_info_a,gid_impr_info_b):
    judge = []
    judge.append(is_time_near(gid_impr_info_a.start_impr_time,gid_impr_info_b.start_impr_time))
    judge.append(is_time_near(gid_impr_info_a.end_impr_time,gid_impr_info_b.end_impr_time))
    judge.append(is_impr_near(gid_impr_info_a.impr_total,gid_impr_info_b.impr_total))
    #judge.append(is_read_impr_ratio_near(gid_impr_info_a.read_impr_ratio,gid_impr_info_b.read_impr_ratio))
    return True if all(judge) else False

def find_a_similar_gid(gid_impr_df_static,gids_impr_df):
    pass


def get_gid_start_impr_time(impr_df,gid):
    return impr_df.iloc[np.where(impr_df["group_id"] == gid)[npWhereIndexListIndex][startIndex],]["impr_time"]
    
def find_part_similar_gids(gid_impr_df_static,gids_impr_df):
    group_id,start_impr_time,end_impr_time,impr_total = GidImprDf(gid_impr_df_static)()
    gid_impr_info_a = GidImprInfo(group_id,start_impr_time,end_impr_time,impr_total)
    part_similar_gids = []
    judge_by_impr_and_time = []
    for row in gids_impr_df.iterrows():
        time = row[pdIterrowsInfoIndex]["impr_time"]
        impr = row[pdIterrowsInfoIndex]["impr_accumu"]
        judge_by_impr_and_time.append(is_time_near(gid_impr_info_a.end_impr_time,time) and is_impr_near(gid_impr_info_a.impr_total,impr))
    for row_index in np.where(judge_by_impr_and_time)[npWhereIndexListIndex]:
        group_id = gids_impr_df.iloc[row_index,]["group_id"]
        end_impr_time = gids_impr_df.iloc[row_index,]["impr_time"]
        impr_total = gids_impr_df.iloc[row_index,]["impr_accumu"]
        start_impr_time = get_gid_start_impr_time(gids_impr_df,group_id)
        gid_impr_info_b = GidImprInfo(group_id,start_impr_time,end_impr_time,impr_total)
        if is_similar_gid(gid_impr_info_a,gid_impr_info_b):
            part_similar_gids.append(group_id)
            
    return list(set(part_similar_gids))


def plot_time_y_relation(plot_data,variable):
    import seaborn
    import matplotlib.pyplot as plt
    fig,ax= plt.subplots()
    fig.set_size_inches(10,8)
    seaborn.pointplot(x=[str(x)[8:12] for x in plot_data['impr_time'].tolist()], 
    y=plot_data[variable],hue = plot_data["group_id"],data=plot_data, join=True,ax=ax)
        
   
    
## 对于重复操作的，可以用循环把这些都写一遍；以对象加以管理？
    
vulgar_impr_total = get_impr_total(vulgar_impr_hour)
high_impr_total = get_impr_total(high_impr_hour)     

# 排序
vulgar_impr_total_sort = vulgar_impr_total.sort_index(by = "impr",ascending=False)
high_impr_total_sort = high_impr_total.sort_index(by = "impr",ascending=False)

# 加入阅读展示比的考虑
vulgar_impr_hour["read_impr_ratio"] = vulgar_impr_hour["read_accumu"]/vulgar_impr_hour["impr_accumu"]
title_impr_hour["read_impr_ratio"] = title_impr_hour["read_accumu"]/title_impr_hour["impr_accumu"]
high_impr_hour["read_impr_ratio"] = high_impr_hour["read_accumu"]/high_impr_hour["impr_accumu"]
vulgar_impr_total_sort["read_impr_ratio"] = vulgar_impr_total_sort["read"]/vulgar_impr_total_sort["impr"] 
high_impr_total_sort["read_impr_ratio"] = high_impr_total_sort["read"]/high_impr_total_sort["impr"] 
# 一个转换会把本来该是int的也转成float;不会，只是在取一行的series时，没法再保持不一样了，会转化为一样的类型
# 先取列，再取位置是个好习惯

# 总阅读展示比对比：
float(sum(vulgar_impr_total_sort["read"]))/sum(vulgar_impr_total_sort["impr"]) # 0.225
float(sum(high_impr_total_sort["read"]))/sum(high_impr_total_sort["impr"]) # 0.101

# 分析
gid = vulgar_impr_total_sort.iloc[4,]["gid"]
gid_impr_df = get_gid_impr_df(vulgar_impr_hour,gid)
end_row_temp = np.where(gid_impr_df["impr_accumu"] < 2300000)[0][-1]
temp = gid_impr_df.iloc[startIndex:end_row_temp,]
similar_gids = find_part_similar_gids(temp,high_impr_hour)
similar_gid_impr_df = get_gid_impr_df(high_impr_hour,similar_gids[1])
plotdata = temp.append(similar_gid_impr_df)
plot_time_y_relation(plotdata,"impr_accumu")




# 找在这个“每日”时间附近达到了对应展示量的
#begin_row = np.where(vulgar_impr_hour["group_id"] == vulgar_impr_total_sort.iloc[0]["gid"])[0][0]




# 注意：很多数学函数都没有办法直接对list使用，需要写个简单循环


#iloc取位置，ix取index
#index
#,hue=temp["holiday"]
#30篇大于100w，695篇大于10w
#.tolist()
#list + set
a.index(3)
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '../common')))

        
# 简洁明快：
# attrs=  {}
# for key in ATTRS:
#    attrs[key] = None
#    self.__dict__.update(attrs)
        
#  self.__dict__[key] = value
# zip(x,y)
        
# 简洁明快的写一个类的初始化，pipeline
# pipeline的模板积累
        
# 看修改功能时会在哪里同时改
basicInfoKeys = ["group_id","total_impr","total_read","first_impr_time","last_impr_time"]

basicMetrics = ["read_impr_ratio","read_impr_ratio_lis","impr_ratio_lis"]       
class dummy:
    def __init__(self):
        metrics = {}
        for key in basicMetrics:
            metrics[key] = None
        self.__dict__.update(metrics)
        
    def __setattr__(self,key,value):
        if hasattr(self,key):
            self.__dict__[key] = value
        else:
            raise Exception("metric not exist")
            
    def run(self):
        # self.read_impr_ratio = 9
        # self.kk = 1
        self.__dict__.update({'a':1})
        return self

[1] * 3
len(list(set(data["group_id"].tolist())))
read_csv:
    columns 随着 header=None
    header = True 被翻译为header = 1，然后就用第二行代替了
    default: header = infer
    header = 0代表第一行
df.drop: 【】传列表代表行数
用行的index名称drop？
drop 名称代表按列名drop，可以用columns取到以后用列位置drop


a = [1,2,3]
sum(a)
a.sum()
import numpy as np
np.array(a).sum()
data['a'] > 0:pandas列
data['a'].astype(int): pandas列
data["a"].values numpy array

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
a = []
for i in range(3):
    a.append([])
    
range(0,9,2)

用instance有继承的好处：
isinstance(np.float64(23.12),float)
type(np.float64(21.23)) == float


from quality_review.util.editor_task_util import in_task_project, create_tcs_task
from quality_review.util.article_retriever import retrieve_article_info_new
info = retrieve_article_info_new(xigua_video_gids, conv=True)
for gid in info:
    create_tcs_task(self.xigua_video_project, gid, info[gid])
    
import sqlalchemy; print(sqlalchemy.__version__)
DJ_CONF=prod python 其实也可进入
DJ_CONF=prod python temp_try_2.py
这样甚至也不行了
返回值

a.append(4) if 1 else b.pop(1)
a.append(4) if 1 else None
a.append(4) if 1 else pass

def apply_to_lis(func):
    def wrapper(some_val):
        # some_val = args.pop(some_val)
        if isinstance(some_val, float):
            return func(some_val)
        if isinstance(some_val,list):
            return [func(val) for val in some_val]
        return some_val
    return wrapper

@apply_to_lis
def now(some_val):
    return round(some_val)
    
def a

        else:
 15                     task_id = j_ret['data']['task_id']
 14                     result_queue.put((group.group_id, int(task_id)))
 13                 count += 1
 12             except Exception as e:
 11                 logging.error("[%s] tcs sender post gid: %s error" % (self.sender_name, group.group_id))
 10                 logging.error(e)

def kk():
    return True if 1==2 else False

class A(object):
    def __init__(self):
        pass
    
    @staticmethod    
    def get_a():
        return 1
    
    def get_b(self):
        return A.get_a()
    
{'a': 1, 'b': 2, 'c': {'cc': 1}}
a.get('c').update({'d':2})
a的深拷贝，浅拷贝与引用实质（可改变）

def get_today_yesterday():
    today = datetime.date.today()
    oneday = datetime.timedelta(days = 1)
    return today, today - oneday

today, yesterday = get_today_yesterday()
url = 'https://tcs.bytedance.net/api/v2/task_results/'
project_id = '1'
url = url + project_id + '/?resolve_time=' + str(yesterday) + ' 00:00,' + str(today) + ' 00:00&per_page=' + str(1000)


counter_client = CounterServiceClient("data", "nlp", "group_profile")

infos_num = defaultdict(lambda :defaultdict(int))
初始化的良方；能不能不用orderdict

infos_num = defaultdict(lambda :defaultdict(int))
 infos_num.keys()
infos_num['a']['b']+=1

from ss_data.counter_service.client_thrift import CounterServiceClient
counter_client = CounterServiceClient("data", "nlp", "group_profile")
2018-05-18 11:10:26,645 gid : 6556550807406248455, features: [0.12, -0.02, -0.14, -0.17, 0.2, -0.22, -0.01, 0.01, 0.06, -0.23, 0.1, 0.15, 0.13, -0.05, -0.02, -0.09, -0.05, -0.01, 0.16, 0.0, -0.08, -0.03, -0.3, 0.19, -0.07, -0.16, 0.02, 0.14, 0.03, -0.04, -0.03, -0.21, 0.18, -0.13, 0.21, -0.05, 0.06, 0.04, 0.1, -0.01, 0.16, 0.03, 0.11, 0.07, 0.0, 0.04, -0.07, -0.28, -0.02, 0.03, 0.07, -0.04, 0.09, 0.14, -0.07, -0.0, -0.03, 0.01, -0.05, 0.0, -0.0, 0.16, -0.08, -0.06, 0.32, 0.32, 0.24, -0.22, -0.23, 0.24, -0.06, -0.38, -0.31, -0.29, 0.3, -0.34, -0.25, 0.1, 0.1, 0.23, -0.13, 0.13, 0.22, 0.25, -0.23, 0.2, 0.08, -0.33, -0.2, -0.23, 0.23, 0.27, 0.3, -0.2, 0.13, 0.21, -0.24, 0.12, 0.25, -0.13, -0.24, -0.26, -0.11, 0.14, -0.26, 0.16, -0.07, 0.27, -0.13, 0.23, -0.36, 0.23, 0.23, -0.01, -0.16, -0.11, 0.12, 0.1, 0.2, 0.21, 0.14, 0.2, 0.12, 0.26, 0.08, 0.14, 0.25, -0.38, 0.0, 0.96, 0.51, 0.0, 0.07]
2018-05-18 11:10:26,814 gid :6556550807406248455, bad_prob :0.2, is_video: 0
2018-05-18 11:10:54,353 6552709959690224141: 6556550807406248455 create task, detail: {"message": "success", "code": 0, "data": {"action": "created", "task_id": "6556753141679784456"}}



def get_yesterday(date = '20180526'):
    day = datetime.datetime.strptime(date, '%Y%m%d')
    oneday = datetime.timedelta(days = 1)
    return day, day - oneday


day = datetime.datetime.strptime提供了date + 分秒级的时间，错误
实际应该只提供天级
datetime().date

for初值写在for循环外！！怎么会犯这么愚蠢而低级的错误

def GetFileNameAndExt(filename):
 import os
 (filepath,tempfilename) = os.path.split(filename);
 (shotname,extension) = os.path.splitext(tempfilename);
 return shotname,extension

Json

ValueError: ValueErr...har 1)',)
ValueError: Expecting property name: line 1 column 2 (char 1)
json.loads("{'a':false}")
str({'is_label':None})
eval(str(a))

pd.concat

retry:None
没取到东西（None can not iterable）
对接口的认识

如果保守了，出来的结果也不对？

dict(zip(a,b))




6556340579456778766
6555634918326534670
6554896037843042830
6552742563818766862
6554287860638286340
6552431836172124679
6556582789783749128
6555386519132045832
6551600917337080327

行的错位

比起能不能得到，能不能得到一样的也很重要。但是该怎么测试呢

OrderedDict({'a':2,'c':55,'b':6})
只看输入顺序，不看排序


6552389139994509838

6551600917337080327

temp_trouble_gids_1000.csv   temp_trouble_gids_12600.csv  temp_trouble_gids_16000.csv  temp_trouble_gids_17600.csv  temp_trouble_gids_19600.csv  temp_trouble_gids_60000.csv  temp_trouble_gids_7800.csv  temp_trouble_gids_9200.csv
temp_trouble_gids_10200.csv  temp_trouble_gids_13000.csv  temp_trouble_gids_1600.csv   temp_trouble_gids_1800.csv   temp_trouble_gids_3400.csv   temp_trouble_gids_6000.csv   temp_trouble_gids_8200.csv  temp_trouble_gids_9400.csv
temp_trouble_gids_11800.csv  temp_trouble_gids_14800.csv  temp_trouble_gids_16600.csv  temp_trouble_gids_19200.csv  temp_trouble_gids_4600.csv   temp_trouble_gids_6200.csv   temp_trouble_gids_8400.csv
temp_trouble_gids_1200.csv   temp_trouble_gids_15800.csv  temp_trouble_gids_17400.csv  temp_trouble_gids_19400.csv  temp_trouble_gids_5200.csv   temp_trouble_gids_7600.csv   temp_trouble_gids_8600.csv


 re.search('a','ab')
 re.match('a','ab')
 re.findall('a','ab')
 
re.match('a','ab')
re.search('a','ab')
re.search(r'a','ab').groupdict()
re.search(r'a','ab').groups()
 m = re.match(r"(..)+", "a1b2c3")
  m = re.search(r'b.','abcdebc')

st = 'aba'  
  st.replace('a','c')
  
  
'title_reo', 'title_bait', 'low_quality_score_v2', 'vulgarity', 'sansu_score', 'title_spammer', 

def dump_py_obj(py_obj, file_name):
    with codecs.open(file_name, 'wb', encoding = 'utf-8') as f:
        pickle.dump(py_obj, f)
    return True

def dump_json_res(some_jsonable, file_name):
    res_str = json.dumps(some_jsonable)
    with codecs.open(file_name, 'wb', encoding = 'utf-8') as f:
        f.write(res_str)
        
规定time_str与date_str的格式？

dict.fromkeys(['a','b','c'],True)
datetime, timestamp 字符串
OrderedDict({'a':'1'})

predict怎么办！！

OrderedDict({'a':1,'b':2}).values()
{'a':1,'b':2}.values()

dict哈希顺序看起来是一样的，OrderedDict也不能改变这一点
b = OrderedDict([('b',1),('a',1)])
b.update({'b':0,'a':1})
b.update({'b':0})
b.update({'a':1})
下面这种传法就能保序

features没问题，tag没问题，ok

化dict的方法

a = {u'a':1, u'b':2}
a['a']

 'a'==u'a'
 '0.78'==u'0.78'
 isinstance(u'0.78',(str,unicode))
 
len(gid_online_diff_mock): 这一部分看看也
 list(set(temp1) - set(temp2))
 import numpy as np
 import pandas as pd
data1 = pd.read_csv('video_high_impr_0610.csv')
data2 = pd.read_csv('test_video_high_impr_0610.csv')
gid_online = data1['object_id']
gid_mock = data2['object_id']
gid_diff = list(set(gid_mock) - set(gid_online) )
gid_online_diff_mock = list(set(gid_online)-set(gid_mock) )
len(gid_online)
len(gid_mock)
len(gid_diff)
len(gid_online_diff_mock)
data3 = pd.read_csv('video_high_impr_before_0611.csv')
data3.columns = ['gid','create_time']
gid_online_all = data3['gid']
gid_real_diff = list(set(gid_mock) - set(gid_online_all) )
pd.DataFrame(gid_real_diff,columns=['gid']).to_csv('video_gid_diff_0610.csv',index = False)

data_combine = pd.merge(data1,data2,on = 'object_id')
time_1 = [datetime.strptime(time,'%Y-%m-%d %H:%M:%S') for time in data_combine['create_time_x'] ]
time_2 = [datetime.strptime(time,'%Y-%m-%d %H:%M:%S') for time in data_combine['create_time_y'] ]
time_delta = [time1-time2 for time1,time2 in zip(time_1,time_2)]
pd.Series(time_delta).quantile(0.1)


for gid,reason_ in reason.items(): group_source.add( get_group(gid).group_source ) if reason_ == '文章类型不符合进审' else None
0.2-0.95 75%在8分钟以内
少的这：
（文章600+，几乎无影响。）
视频3000+，
多的这：lock
目前逻辑 + 怎么追查

## gid_real_diff 仍然有4572的diff
for gid in unvalid_gids: aa.update({gid: get_group(gid).group_source == 7})
index 找到
再keys定位妙招

from datetime import datetime
datetime.strptime('2018-06-10 00:00:00','%Y-%m-%d %H:%M:%S')
time_1 = [datetime.strptime(time,'%Y-%m-%d %H:%M:%S') for time in data_combine['create_time_x'] ]
time_2 = [datetime.strptime(time,'%Y-%m-%d %H:%M:%S') for time in data_combine['create_time_y'] ]
time_delta = [time1-time2 for time1,time2 in zip(time_1,time_2)]
pd.Series(time_delta).describe()

data2.iloc[np.where(data2['object_id'] == 6564985388463555076)[0][0],]

import cPickle as pickle

/data00/home/zhangyu.95/201806_working/article_audit 
txt还是有办法搞上去的
依赖环境越少越好，然而
分别有3936和794没进
 
 1498 去除已经进过的;
 794
 
 6-11的
 视频的形成同一套方法
 像峰池一样搞main
 
 状态。
 gid
 test用绝对路径即可
 
data2.iloc[np.where(data2['object_id'] == 6564985388463555076)[0][0],]
两边select一下

def mock(choice = None):
    if not choice:
        print 1
    
debug 分支可以推上去
但还是会有漏掉的地方？
应该不会，debug分支的代码哪些要合到主枝上
！！可以debug一次就针对debug的内容进行保存与分析，注意分支命名
需要完整debug的时候再加工具。

logging.warning('Reasons are %s' % {reason_type:reason.values().count(reason_type) for reason_type in set(reason.values())})    |~
import cPickle as pickle                                                                                                        |~
pickle.dump(reason, '/data00/home/zhangyu.95/201806_working/article_audit/reason.txt')

datetime.strptime('2018-06-01 00:00:00','%Y-%m-%d %H:%M:%S')

至少想过不要再想一遍
debug涉及子函数的不同参数，自己的内部插入等

 return os.environ.get('SCRIPT_NAME')

article.
crawl/article.py

/data00/home/zhangyu.95/201806_working/article_audit_abandon/src/model_tools


4753  
289

超时


两种可能的解决方式，去选用。

import pandas as pd
data = pd.read_csv('test_article_0611.csv')
data = data.drop(data.columns[ [0,1,2,3] ], axis=1)
aa = data.values
len(aa)
np.any(np.equal(aa, None))


nan 用 0 替换
numpy中nan_to_num可以解决
直接的nan不能检测nan数量

给出了numpy.ndarray的y_pred，就更为灵活了（测试时）
接口与定义


article_audit_abandon/model_tools
article_audit_online_v2

test_auc





 import pandas as pd
  1     import argparse
  2     parser = argparse.ArgumentParser()
  3     arguments = ['-gid','-gids','-filt','-gid_file']
  4     for arg in arguments:
  5         parser.add_argument(arg)
  6     args = parser.parse_args()
  7
  8     # print type(args) ## 以及args的转化方式
  9     # print args.__dict__ #dict可妙用
 10     if args.gid:
 11         gids = [int(args.gid)]
 12     if args.gids:
 13         gids = eval(args.gids)
 14     if args.filt:
 15         filt = eval(args.filt + '()')
 16     if args.gid_file:
 17         default_path = '/data00/home/zhangyu.95/201806_working/article_audit/test_gid_file/'
 18         default_id = 'gid'
 19         gids = list(pd.read_csv( default_path + args.gid_file )[default_id])
 20
 21     filtered_gids = filt.filter_gids(gids)
 22     unvalid_gids = list(set(gids) - set(filtered_gids))
 23     print 'gid_left:', filtered_gids[0:5]
 24     print 'gid_unvalid', unvalid_gids[0:5]
 25     import pdb;pdb.set_trace()

dumps
loads
kaf

convert

str(u'1')
str(u'AA')
str(u'啊')


getattr在模块化中的使用：
置顶的class或者什么的import后可以直接getattr用


str(u'啊啊')
'啊啊'.decode('utf-8')
json.dumps(1)
json.dumps('a')
'aa'.decode('utf-8')
u'aa'.decode('utf-8')
都作为unicode传出去，否则中文字符会出问题

有\n怎么去
\\n -> 空格

strr = json.dumps({'aa\nbb':0.3})
'\\n' in strr
strr.replace('\\n','  ')
\\n
'\\n'
isinstance('a',str)
isinstance(u'a',str)
isinstance(u'a',unicode)

json dumps的时候 \和引号都会翻倍
超多转义字符，针对引号

b = {x:y for x,y in a.items() if y > 4}

encode('utf8') 与 decode('utf8')

getattr相当于.
当getattr找不到属性时，就触发内建的__getattr__
一般的__getattr__内容是？？！

__
_ 
如何善用。
getattr不了。

class A(object):
    def _a(self):
        print 1
    
    def b(self):
        getattr(self,'_' + 'a')()
        
for line in file('try_file.conf'):
    aa.append(line)
    
同时继承两个类的特性？
class A(object):
    def __init__(self):
        self.a = 1
        self.b = 2
    
class B(A):
    def __init__(self,val):
        super(B, self).__init__()
        self.c = val
        
继承父类函数的部分功能？

    
拆成几个简单模块dump，到时候再合起来
with open('try.csv','wb') as final_data:
    reader = csv.reader(final_data)
    row = reader.next()
    
tag对应方法不一样
特征抽取来源不一样
dump tag有什么一样之处？有。直接查看tcs审核结果
dump feature 如果使用抽特征服务 + 我们的dump方式，可以考虑让它来源一致。
因此dump数据也可以结构化

维护log， dump 和 train
    
a = 1
3 <= a <=4

pd.Series(True) | pd.Series(False)
pd.Series(True) & pd.Series(False)
pd.Series([True,False]) | pd.Series([False,True])
