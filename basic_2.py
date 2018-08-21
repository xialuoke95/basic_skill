# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:37:03 2017

@author: zy
"""
#----------------------- 重要快捷键
#-- 清屏 ctrl+L

#--------------------------------- 
#-- 清空工作空间  
reset

reset in
reset out
#-- 清屏
%clear
#-- 强行停止：点击红框，没别的好办法

#-- print不换行的办法

#-- 读取文件的编码问题
stopword = pd.read_table("originData/stopword.txt",header = None,encoding = 'GB2312')
#--报错

#--查看默认编码的可能方式

stopword = pd.read_table("originData/stopword.txt",names = "stopwords")
#-- "stopwords"作为一个字符串向量，被视为字符的列表

stopword2 = pd.read_table("originData/stopword2.txt",names = ["stopwords"],encoding = 'gb2312')
#-- 文件无法含有中文名；内部含有中文也需要gb2312进行解码

#--改编码以后还要改回来
stopwords = pd.read_table("originData/stopword.txt",names = ["stopwords"])
stopwords = {}.fromkeys([ line.rstrip() for line in open('originData/stopword.txt',encoding = 'utf-8') ])
#--open的用法
path =  "history_data\obsnapshots_CST_20171202_01-48-05.json"
file= open(path,"rb")
open('originData/train.tsv',encoding = 'GB2312')

#--中文也常常utf-8
for line in open('originData/train.tsv',encoding = 'utf-8'):
    line = line.strip()
    
for line in open('originData/train.tsv',encoding = 'gb2312'):
    line = line.strip()

#--或者用codecs打开文件
with codecs.open('1.json','r','utf-8') as f:
     temp = f.read()


#--列名？元素名？
stopword = pd.read_table("E:/A.Project/Python_project/2017_CCF_textmining/originData/stopword.txt",header = None)
stopword[0]
stopword[0][0]
stopword['0'][0]

stopword = pd.read_table("originData/stopword.txt",names = ["stopwords"])
stopword[0][0]
stopword['stopwords'][0]

!cat originData/stopword.txt
#-- 会看所有的，而且格式不对？

#-------------------------- 变量类型：弱类型语言
#-- 查看变量类型：
1 有哪些类型
2 
type()
isinstance()
#-- 可判断继承关系
dir()
#-- 可查看对象类型的属性和方法

#-- 列表与数据框的互相转换

a = ['cheche',1,2]
pd.DataFrame(a)
b = pd.DataFrame(a)
b[0][0]
b[0][1]
b[0][1] + b[0][2]
a.to_csv('a.tsv')
a[1:3]
b[1:3]
左包括，右边不包括
b = pd.Series(a)
b[1]
 
b.to_csv('b.tsv',sep = '')

#dtype:object?

#--怎么按行一行一行读入？ 只读前几行？ 包括R里怎么做。
#--中文编码怎么在保存的时候调整的？怎么用命令识别？



#-- 查看当前工作目录
import sys
import os
os.getcwd()

#-- 查找当前Python依赖路径(搜索路径)
import sys
sys.path
sys.path.append()

#-- 
.to_csv()

#--

[9996] * 3

#--
try:
    a = 1
except:
    pass ## 什么都不做
    
set([1,2,3,3,4]).pop()
a = ["if","then","extra"]
" and ".join(a)

a = [1,2,3,4]
a[:]
e = [{'task_id': i,'gid': j} for i,j in [(2,4),(3,7),(6,5)]]
e.get('task_id')
e.get('task_id'，0)
e[0].get('task_id')

cnt = {}
cnt.setdefault("a",0)
cnt
cnt.setdefault('b',0)

cnt.setdefault('a',1)

cnt.get('a',0)
cnt.get('a',1)
cnt.get('c',0)

a = '3'
int(a)
a = '680730170480138029481830'
int(a)

a = [1,2,3]
b = []
b = [1,2,3]
b.append(a)
b
b.extend(a)
b

filter(a>2)
a = [1,2\
,3,4]
dict([("a",[])])

b['a'].append(1)

map(int,['356','789'])

filter(lambda a:a>3,[1,2,3,4])

for i in [1,2,3]:
    print i    
i
循环后解释器会存下最后这个变量的值
a = [1,2,]
a.remove(1)

a = [1,2]
b = [2,3]
c = [a,b]
for i in c:
    i.pop()
    
    
0>None
    
print "Origin push group is %d, with max ratio %d ; same sim group is %d, with max ratio %d." \
% (4,5,6,7)

name_level_project_map = {
    '科技2级': 6483339288216863245,
    '体育2级': 6482601214855873037,
    '其他2级': 6483340295797735949,
    '科技345级': 6495517853653926414,
    '体育345级': 6495515472232972813,
    '其他345级': 6495518878892818958,
    '汽车': 6482600677716525581,
    '娱乐': 6483802141008855566,
    '财经': 6483802586385220110,
    '三农': 6504548891776516622,
}
tcs_name_level_project = ['tcs-' + str(queue_id) for queue_id in name_level_project_map.values()]


from pyutil.program.conf import Conf
nsq_conf = Conf("push_client_debug.conf")
addr = nsq_conf.nsq_addr.split(',')
dir(nsq_conf)
很奇妙的结果

datetime.datetime.now()
time.time()


for i in [1,2,3]:
    print i   
i



lis = [(1,2),(3,4),(1,7)]
for a in lis:
    if a[0]==1:
        lis.remove(a)
        
lis
a

lis = [(1,2),(3,4),(1,7),(2,9)]
for a in lis:
    if a[0]==1:
        lis.remove(a)
lis
a
        
for a in lis:
    if a[0]==1:
        lis.remove(a)
    else:
        pass     
a
lis
尽量else:pass不然会出现无法预料的结果

round(0.0082353,4)


a = {"recent_user_action": ["3:12_69351701700", "3:12_69892169257", "3:12_71302344587", "3:12_97339985997", "-2:12_50322774714", "3:12_61993475619", "3:12_79026259572", "3:12_68631020121", "3:12_95589188160", "3:12_93576512621", "3:12_52199096121"]}
json.dumps(a)
import json
json.dumps(a)
b = json.dumps(a)
a = json.loads(b)
a
a = {'aa':1,'bb':{'ccc':1}}
c = a['bb']
c
c.update({'ab':1})
c
a

对于这种字典内取东西，如果不deepcopy就随意改变原值，是危险的！
int(678.0)
int('678.0')

pb序列化与打redis，神奇复现
首次get是整数，尽管pb定义了浮点
看看get以后传回情况。
但是既然改成了浮点，就不应该仍假定为整数了。整数在这种场景下也不好用。

from first import first
first([0, None, False, [], (), 42])

a = A()
a
a.a
a.set()
a
a.a
b = A()
c = b
c.set()
c
c.a
b.a
自己写的对象，如果有可改变的地方，又不是deepcopy而是直接赋值，也有同样危险

1）多重继承时**args传什么
2）如何拆分
3）多重**args的使用方法

class A(object):
    a = 1
    b = a + 1
    
    def __init__(self):
        pass
        
class As(A):
    def __init__(self, **args):
        super(As, self).__init__()
        if 'b' in args:
            self.a = args['b']
        
class Ass(As):
    def __init__(self, **args):
        super(Ass,self).__init__(**args)
        
必须用args的时候：

article for article .gid in ret_set
???这会执行出什么结果

a = {}
aa = a.get('a')

'<{:30s}>'.format('a') 最长30的格式化字符串

def a(b = True,c):
    pass

def a(b,c):
    pass

@abstractmethod

@classmethod

@staticmethod

@property

这种框架性的代码

shi基类的单元测试方法

class A:
    a = 1
    
import types
type(a) == types.InstanceType

class A(object):
   a = 1
a = A()
type(a) == types.InstanceType

新类，只能指定是一个特定类；因为所有东西都是类

from ss_data.profile.smart_group_store import ThriftGroupProfileStore
ThriftGroupProfileStore(
    service_name='data.profileservice.nearline',
    identity="anti_spam",
)

get_profiles(gids, [("g_quality_level", 0)] )

uni = []
for line in file('filtered_gids_log'):
...     uni.append(line.split(' ')[11])
 bb = set(uni)
 
mget_articles_from_db2_with_chunk([6581429598137876999], need_content=False,caller = 'data.nlp.pull_back')




class A(object):
    def __init__(self,a):
        if a:
            self.a = a
        
    def __getattr__(self,key):
        return 0

a.get('b')
.get的时候调用了__getattr__内建方法，发现call不动

class A(object):
    def __init__(self,a):
        if a:
            self.__dict__['a'] = a
        
    def __getattr__(self, key):
        return 0
    
    def __setattr__(self, key, value):
        self.__dict__[key] = 1
  
  
    
a = A(3)
a.a
getattr(a,'a')
b = A('')
b.a
a = A(3)
b.a
a.a

getattr: 有值时并不调用内建函数，直接取相应值

__setattr__ : 保证了A.a类型的赋值 和 setattr()两种赋值
self.__dict__[key]的赋值不受__setattr__影响    
'media_id'
更新

#  错误用法
def __setattr__(self, name, value):
    self.name = value
    # 每当属性被赋值的时候(如self.name = value)， ``__setattr__()`` 会被调用，这样就造成了递归调用。
    # 这意味这会调用 ``self.__setattr__('name', value)`` ，每次方法会调用自己。这样会造成程序崩溃。

#  正确用法
def __setattr__(self, name, value):
    self.__dict__[name] = value  # 给类中的属性名分配值
    # 定制特有属性
    
    
不调用的话 能否绕过抽象方法？
string[0:3]

import logging
logging.getLogger('aaa')
aa = logging.getLogger('aaa')
aa.warning('1')
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
aa.addHandler(sh)
aa.warning('1')
aa.debug('1')
aa.info('1')
aa.setLevel(logging.DEBUG)
aa.info('1')
aa.debug('1')

log 与 handler的级别关系
handler还能随时间变化而变化
函数无先后而class有先后


"{} instance with\n\t".format(self.__class__.__name__
) + '\n\t'.join(['{:20s} = {}'.format(key, getattr(self, key), None)

\n在logging里不算一行

'{:2s}'.format('abcde'): 不够补足，但长了会更长
'{:7s}'.format('abcde')
\n\t
 

getattr与__getattr__理清:
__getattr__()是仅当属性不能在实例的__dict__或它的类(类的__dict__),或父类其__dict__中找到时，才被调用。


https://openhome.cc/Gossip/Python/PropertyNameSpace.html 超类的__dict__不被更改
没有人可以阻止self.__dict__形式的调用，严禁。
__slot__阻止.__dict__?
https://zhuanlan.zhihu.com/p/26488074 魔法方法

str与repr
%s % ； {}。format：str
直接打是repr
有repr全repr
有str不能搞定直接打对象这一需求



把convert与model
剥离，抽出来
git stash drop 

@property
show_me_the_util

单元测试主要是为了给大家看工具怎么用？

aa = A(3)
字典对应到同一个对象的恶果？

保持其他不变
gid

import time
start = time.time()
a = {}
for i in range(500):
    a.update({i:deepcopy(aa)})

import time
start = time.time()
b = {}
for key,val in a.items():
    b.update({str(key):val})
print time.time()-start

class B(object):
    def __init__(self):
        pass

    def __setattr__(self, key, value):
        self.__dict__[key] = 1
  

hasattr来调用

hasattr(aa, '__copy__')
getattr(aa, '__copy__')

线程等
def kkk(a,b):
    print (1)
    
 kkk(a = 3, b = 4,)

aa = 1
eval('aa')
k = eval('aa')
eval('aa') = l
    
class A(object):
    def __init__(self,a):
        if a:
            self.a = a
        
    def __getattr__(self,key):
        return 0

    def __copy__(self):
        newone = type(self)(1)
        newone.__dict__.update(self.__dict__)
        return newone

    def __deepcopy__(self, memo):
        return self.__copy__()
  
B()
a的分层次 
一个lpush最多push几个东西？   
 
copy只copy一层
        return self.__init__(self.a)
    
    self.dict.update;dir(self)
    
把跟dir相关的打出来；mixin的情况下取__dict__是否够？dir中的其他属性？
deepcopy!
要知道不放在self.__dict__ 中放在哪里
得搞
dir mixin

getattr(aa,'__reduce_ex__')(2)
reduce_ex的神奇属性
只能重写__deepcopy__

  newone = type(self)()
  newone.__dict__.update(self.__dict__)
  return newone

关键是__getattr__之后__getstate__方法遭到了重写
pickle
copy与deepcopy

aa
bb = [aa]
cc = [bb[0]]
bb
cc
cc = [bbb for bbb in bb]
cc

还是得重写deepcopy！

g.group_id for g in groups是奏效的

a = [a for a in A]

a = [A(3),A(4),A(5)]
def run(groups):
    ret = []
    for group in groups:
        if group.a > 4:
            group.a = 2
        else:
            ret.append(group)
        
    return ret

def run(groups):
    for group in groups:
        if group.a > 4:
            group.a = 2
        else:
            group.a = 0
        
    return groups

a = run(a)

新分配空间
主要是指向的list位置发生变化, 过滤了一些之后；指向对象本质未变；

访问时，原先位置的对象还在不在？应该没用了的话就不在了吧，而且也许也就存一些指针

extend的时候本质是如何扩展？

def aaa(a,**params):
    pass

两个基本类型；Nan，如何做到表现和None一样而 和NA不一样？

aa = ['source_icon', 'item_status', 'video_infos', 'has_m3u8_video', 'keywords', 'has_mp4_video', 'pgc_ad', 'article_group_url', 'title', 'source', 'natant_level', 'own_group', 'share_count', 'is_ad', 'list_play_effective_count', 'categories', 'repin_count', 'display_status', 'level', 'digg_count', 'cover_image_infos', 'group_id', 'middle_image', 'subject_display_status', 'has_dynamic_gif', 'pgc_id', 'ad_type', 'create_time', 'book_info', 'article_sub_type', 'label', 'content', 'group_source', 'is_key_item', 'forum_id', 'item_id', 'good_voice', 'max_comments', 'language', 'display_url', 'region', 'content_cards', 'has_gallery', 'modify_time', 'content_cntw', 'detail_mode', 'impression_count', 'image_list', 'group_status', 'creator_uid', 'original_media_id', 'city', 'bury_count', 'web_article_type', 'review_comment', 'comment_count', 'lang', 'media_id', 'go_detail_count', 'detail_play_effective_count', 'visibility', 'was_recommended', 'tc_head_text', 'thumb_image', 'external_visit_count', 'image_detail', 'ban_action', 'review_comment_mode', 'has_inner_video', 'has_image', 'ban_comment', 'abstract', 'middle_mode', 'is_original', 'ban_bury', 'article_type', 'tag', 'id', 'optional_data', 'mobile_url', 'has_gif', 'has_video', 'article_url', 'display_mode', 'composition', 'publish_time', 'author_delete', 'tag_id', 'pgc_article_type', 'display_flags', 'display_type', 'gallery', 'detail_source', 'url', 'web_display_type', 'image_infos']

class BB(object):
    def __init__(self,**params):
        self.a = params.get('a')
    
    def check(self):
        if hasattr(self,'a'):
            return True
        else:
            return False
        
BB().check()

kk = {}
kk.update({'a':2}) if a==2 else None

a = a+1
没有更改，不再指向
A = {'a':1}
def drop(k,name):
    ret = {}
    k.pop(name)
    return ret.update(k)
def drop(k,name):
    ret = {}
    k.pop(name)
    return k
A = drop(A,'a')
drop(A,'a')
A = drop(A,'a')
本身改变了，赋值地址也改变了。
return的本身也是指向地址

lis = [{'b':2},{'c':3}]
[a.update({'a':1}) for a in lis]
 kk = [a.update({'a':1}) for a in lis]
 吃返回值，但是lis同时也会被改变
 
 def func(a):
     a = a + 1
     return a
 
def func(A):
    A = A.get('a')
    return A

内部赋值，还是不会变的。变量名在内部暂时重名

'a,c,d,b'.split(',',2)
只分前两个；乃至只分后一个：妙啊。

订房间

from crawl_data.domain.article.functool import close_django_old_connections
@close_django_old_connections                                                                                                       
def get_hot_gids(gids, with_l2=True):                                                                                        
    channel_hots = list(ChannelHotArticle.objects.filter(group_id__in=gids).order_by('create_time'))
    return channel_hots

hot_gids = set()
for hot in channel_hots:
    if hot.level == 1:
        hot_gids.add(hot.group_id)
         

常数在哪里看   
expire_time__gte=datetime.now()
from pyutil.program.conf import Conf
CONF = conf = Conf(os.path.dirname(os.path.abspath(__file__)) + '/conf/db_prod.conf')

为什么它那样conf就行我就不行？看

dict get, 非dict getattr

Traceback


select name from channel_conf where id = 3189398996
select * from channel_conf where name = '热点'
select id from channel_conf where name = '央媒'


select * from channel_conf where id = 72115010420
select * from channel_conf where id = 3189398996
select * from channel_conf where id = 3189398999

group.webdb_info