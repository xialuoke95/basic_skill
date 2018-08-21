#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 10:39:51 2018

@author: zy
"""
def delta(lis):
    delta_lis = []
    for i in range(1,len(lis)):
        delta_lis.append(lis[i] - lis[i - 1])
    return delta_lis

def distance(value1,value2):
    return abs(value1-value2)

## 返回min(near)以备检查
def find_near_index(lis,value):
    near = [distance(val,value) for val in lis]
    return near.index(min(near)),min(near)

def is_chosen_by_time(time,start_time,time_interval):
    return True if (time - start_time) % time_interval == 0 else False

## 目前不检查错误
def get_impr_df(impr_df,gid):
    return impr_df.iloc[np.where(impr_df["group_id"] == gid)[npWhereIndexListIndex],]

# 犯了很多同样的错误：self.__impr_df再调self的私有方法
class GidImprDf(object):
    def __init__(self,gid_impr_df):
        self.__impr_df = gid_impr_df
        self.__need_columns = needColumns
        self.__check()            
        self.__group_id = self.__get_group_id_with_check()
        self.__default_time_interval = 10 * 60
        self.__last_row_index = self.__total_row_index = self.__get_total_row_index()

## 至少两行，有相应信息列        
    def __check(self):
        if not isinstance(self.__impr_df,pd.core.frame.DataFrame):
            raise Exception("Illegal Input: not dataframe")
        check_row = [self.__nrow() > 1]
        check_col = [col in self.__impr_df.columns for col in self.__need_columns]
        if not all(check_row + check_col):
            raise Exception("Illegal Input:lack row(raw info) or columns") 
        return
    
    def __get_group_id_with_check(self):
        group_id_lis = list(set(self.__impr_df.get("group_id").tolist()))
        if not len(group_id_lis) == 1:
            raise Exception("Illegal Input:Gid not unique") 
        return group_id_lis[Start_Index]
    
    #def __get_default_time_interval(self):
    #    time_lis = self.__impr_df.get("impr_time").tolist()
    #    time_delta = list(set(delta(time_lis)))
    #    if not len(time_delta) == 1:
    #        raise Exception("Illegal Input:time interval not unique")
    #    return time_delta[Start_Index]
        
    
    def __is_sort_by_time(self):
        pass
    
    def __sort_by_time(self):
        pass
    
    def __nrow(self):
        return self.__impr_df.shape[pdShapeRowIndex]
    
    def __get_total_row_index(self):
        return self.__nrow() - 1
    
    def __get_col_with_colname(self,colname):
        if colname not in self.__impr_df.columns:
            raise Exception("col not exist")
        return self.__impr_df.get(colname).tolist()
    
    def __get_impr_col(self):
        return self.__get_col_with_colname("impr_accumu")
    
    def __get_read_col(self):
        return self.__get_col_with_colname("read_accumu")

    def __get_imprtime_col(self):
        return self.__get_col_with_colname("impr_time")
        
    ##
    def __get_row_index_pair_by_time(self,expect_start_time,expect_end_time):
        time_lis = self.__get_imprtime_col()  
        start_time_index, start_time_diff = find_near_index(time_lis,expect_start_time)
        end_time_index, end_time_diff = find_near_index(time_lis,expect_end_time)
        return start_time_index,end_time_index
    
    def __gen_start_end_time_default(self,minutes_from_first_impr):
        return self.get_first_impr_time(),self.get_first_impr_time() + minutes_from_first_impr * MIN_TO_SECOND
        
    # 怪怪的; 以及这两个函数的公有部分如何精简
    # 而且目前time_interval只能输默认interval的整数倍
    # 向后累计次数保持固定，但是累计出的东西可能有点问题，比如2030，2130，中间漏了一次2100的话
    def __get_accumu_lis_with_colname(self,colname,minutes_from_first_impr,minutes_interval):
        lis = []
        col = self.__get_col_with_colname(colname)
        time_col = self.__get_imprtime_col()
        time_interval = minutes_interval * MIN_TO_SECOND
        accumu_len = time_interval / self.__default_time_interval
        
        start_time,end_time = self.__gen_start_end_time_default(minutes_from_first_impr)
        start_time_index,end_time_index = self.__get_row_index_pair_by_time(start_time,end_time)

        for index in range(start_time_index, end_time_index):
            time = time_col[index]          
            if is_chosen_by_time(time,start_time,time_interval): 
                val_lis = col[index:(index + accumu_len)]
                interval_val = sum(val_lis)
                lis.append(interval_val)  
        return lis
    
    def __get_sample_lis_with_colname(self,colname,minutes_from_first_impr,minutes_interval):
        lis = []
        col = self.__get_col_with_colname(colname)
        time_col = self.__get_imprtime_col()
        time_interval = minutes_interval * MIN_TO_SECOND
        
        start_time,end_time = self.__gen_start_end_time_default(minutes_from_first_impr)
        start_time_index,end_time_index = self.__get_row_index_pair_by_time(start_time,end_time)

        for index in range(start_time_index, end_time_index):
            time = time_col[index]
            if is_chosen_by_time(time,start_time,time_interval): 
                lis.append(col[index])  
        return lis
    
# if else one line只能返回一个值，无法返回一个操作    
    ### 外部接口
    def get_group_id(self):
        return self.__group_id
    
    def get_total_impr(self):
        return self.__get_impr_col()[self.__total_row_index]
            
    def get_total_read(self):
        return self.__get_read_col()[self.__total_row_index]
    
    def get_first_impr_time(self):
        return self.__get_imprtime_col()[Start_Index]
    
    def get_last_impr_time(self):
        return self.__get_imprtime_col()[self.__last_row_index]
    
    # 时间以分钟级别为单位
    # 调取时需提供想要的时间跨度与时间间隔   
    def get_imprtime_lis(self,minutes_from_first_impr,minutes_interval):
        return self.__get_sample_lis_with_colname("impr_time",minutes_from_first_impr,minutes_interval)
    
    def get_impr_accumu_lis(self,minutes_from_first_impr,minutes_interval):
        time_impr = {'time':self.get_imprtime_lis(minutes_from_first_impr,minutes_interval),
         'lis':self.__get_sample_lis_with_colname("impr_accumu",minutes_from_first_impr,minutes_interval)}
        return time_impr
    
    def get_read_accumu_lis(self,minutes_from_first_impr,minutes_interval):
        time_read = {'time':self.get_imprtime_lis(minutes_from_first_impr,minutes_interval),
         'lis':self.__get_sample_lis_with_colname("read_accumu",minutes_from_first_impr,minutes_interval)}
        return time_read
    
    def get_impr_in_interval_lis(self,minutes_from_first_impr,minutes_interval):
        time_impr_in_interval = {'time':self.get_imprtime_lis(minutes_from_first_impr,minutes_interval),
         'lis':self.__get_accumu_lis_with_colname("impr_count",minutes_from_first_impr,minutes_interval)}
        return time_impr_in_interval       
    
    def get_read_in_interval_lis(self,minutes_from_first_impr,minutes_interval):
        time_read_in_interval = {'time':self.get_imprtime_lis(minutes_from_first_impr,minutes_interval),
         'lis':self.__get_accumu_lis_with_colname("read_count",minutes_from_first_impr,minutes_interval)}
        return time_read_in_interval  
    
# 1）只修改basicinfo的工厂？   2）开始接触到设计模式 3)不用eval的办法
    def get_basic_info(self):       
        basic_info = {}
        for key in basicInfoKeys:
            try:
                val = eval('self.get_' + key + '()')
            except:
                raise Exception("get basic info method not exist") 
            basic_info.update({key:val})
        return basic_info