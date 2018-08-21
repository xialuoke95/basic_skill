#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 13:48:12 2018

@author: zy
"""

31     sended_gids = session.query(PushAuditCandidate.group_id).filter(
 30         PushAuditCandidate.create_time > datetime.datetime(now.year, now.month, now.day),
 29         PushAuditCandidate.group_id.in_(gids),
 28         PushAuditCandidate.status == PushAuditCandidateStatus.sended,
 27         PushAuditCandidate.audit.is_(None),
 26     ).all()
这样得到的item【0】就是gid？


def group:
    group = 1
    return group


[{'task_id': i,'gid': j} for i,j in [(2,4),(3,7),(6,5)]]

push_db = DAL(host=c.ss_pushdb_read_host,
                            port=c.ss_pushdb_read_port,
                            user=c.ss_pushdb_read_user,
                            passwd=c.ss_pushdb_read_password,
                            name=c.ss_pushdb_name)

"select id,group_id from app_alert_rule use index(end_time) order by id desc limit 10"
a = push_db.execute("select id,group_id from app_alert_rule use index(end_time) order by id desc limit 10")
vim 在正常模式 临时进入插入模式

是获得多种对象的极佳方式
def get_step_name(obj):
 10     if type(obj) is types.FunctionType:
  9         return obj.__name__
  8     elif type(obj) is types.InstanceType:
  7         return obj.__class__.__name__
  6     elif type(obj) is types.MethodType:
  5         return obj.im_self.__class__.__name__ + "." + obj.im_func.func_name
  4     else:
  3         return str(obj)
  
  pack-121a4de3c31c23553a9705fba7743c5372a3f5d8
  