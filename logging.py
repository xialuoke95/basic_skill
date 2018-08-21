#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 16:16:42 2018

@author: zy
"""

import logging
logging.warning('add_agroup_attr')

class PSMLoggingFilter(logging.Filter):
    def filter(self, record):
        import pdb;pdb.set_trace()
        return not record.getMessage().startswith('add_agroup_attr')
## str class ##

logging.root.addFilter(PSMLoggingFilter)
logging.root.removeFilter(PSMLoggingFilter)

logging.root.addFilter(PSMLoggingFilter())
logging.warning('3')

'333'.startswith('3')


a.strip()
去掉头尾空格
strip与split都是不改变本体的函数
改变本体的只有针对容器，即list、map的