# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd


# 用个人信息登录
def login():
    params = {"username": "ruc:<学号>",  # 这里一定要加ruc:
              "password": "<微人大密码>",
              "remember_me": "false",
              "redirect_uri": "/",
              "twofactor_password": "",
              "twofactor_recovery": ""}

    session = requests.session()

    s = session.post("http://v.ruc.edu.cn/auth/login", json=params)
    # 必须要先请求一次，使进程先获取除了username的cookies（为了后面的选择身份）
    s = session.get(
        "http://app.ruc.edu.cn/idc/education/selectcourses/resultquery/ResultQueryAction.do?method=forwardAllQueryXkjg")
    return s, session


# 向全校课程表界面POST表单获取课程信息
def getClassPage(page, nclass, session, cookies):

    params_100 = {"method": "allJxb",
                  "isNeedInitSQL": "true",
                  "xnd": "2017-2018",
                  "xq": "1",
                  "condition_xnd": "2017-2018",
                  "condition_xq": "1",
                  "pageNo": str(page),  # 页码
                  "pageSize": str(nclass)}  # 每业显示课程数

    return session.post("http://app.ruc.edu.cn/idc/education/selectcourses/resultquery/ResultQueryAction.do", params_100,
                     cookies=cookies)


# 解析html获取上课的时间地点
def get_class_info(html, output):
    bsObj = BeautifulSoup(html.text, 'lxml')
    for c in bsObj.findAll("div", {"id": "detail"}):
        tmp = c.get_text().strip()
        if "2." in tmp:  # 处理存在于一条的多个上课信息
            for i in re.split("2\.|3\.|4\.|5\.", tmp):
                i = i.replace(" ", "").replace("1.", "").replace("周次：", "")
                result = re.split(",|\u3000", i)
                result = pd.DataFrame(result).transpose()
                output = pd.concat([output, result], ignore_index=True)
                #print(result)
        else:
            tmp = tmp.replace(" ", "").replace("1.", "").replace("周次：", "")
            result = re.split(",|\u3000", tmp)
            result = pd.DataFrame(result).transpose()
            output = pd.concat([output, result], ignore_index=True)
            #print(result)
    return(output)


# 利用cookie识别身份
# （由于具有本科和研究生俩个身份，不指名的话会要求选择身份，无法进入全校课程表）
cookies = {"username": "<学号>"}

s, session = login()  # 登录

output = pd.DataFrame()

# 循环获取28页课程信息，每页100条
for p in range(1, 28):  
    s = getClassPage(page=p, nclass=100, session=session, cookies=cookies)
    output = get_class_info(html=s, output=output)  # 更新output
output.columns = ['周数', '星期', '节数', '教学楼', '教室']

print(output)

output_copy = output.copy()
# 获得了3156条上课地点数据（有的一条内有多个上课时间地点）