# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd


# 利用登陆后的cookies跳过验证码登录
def masterlogin(cookies):
    session = requests.session()
    session.get("http://portal.ruc.edu.cn/ypy/main.jsp", cookies=cookies)
    return session


# 解析课程选择页获得学院列表
def get_school_list(session, cookies):
    s = session.get("http://portal.ruc.edu.cn/ypy/checkCourses_main.do", cookies=cookies)
    bs = BeautifulSoup(s.text, 'lxml')
    school_list = {}
    for i in bs.find("select", {"name": "yxsdm"}).option.next_siblings:
        if i == "\n": continue  # 跳过那些换行符
        school_list.setdefault(i["value"], i.get_text())
    return school_list


# 通过POST获取指定学院课程的html代码
def get_school_courses_html(scode, sname, session, cookies):
    ckcors = {"yxsdm": scode,
              "kcnd": "2017-2018",
              "kcxq": "1",
              "xh": cookies["renmin_university_ypy_username"],
              "yxsmc": sname,
              "dpwrid": "32"}

    s = session.post("http://portal.ruc.edu.cn/ypy/checkCourses.do", data=ckcors, cookies=cookies)
    return s.text


# 解析html获取课程信息
def get_class_info(html):
    bs = BeautifulSoup(html, 'lxml')
    teacher = []
    class_name = []
    credit = []
    class_time = []
    room = []
    first_class = []

    for i in bs.findAll("td", {"width": "7%"}):  # 授课教师
        if i.get_text().strip() != "授课教师":
            teacher.append(i.get_text().strip())

    for i in bs.findAll("td", {"style": "text-align:left;"}):  # 课程名称
        class_name.append(i.a.get_text().strip('[|]| '))

    for i in bs.findAll("td", {"width": "6%"}):  # 学分和开课人数（学分没有超过4的，开课人数没有低于5的）
        tmp = i.get_text().strip()
        try: int(tmp)
        except: continue
        if int(tmp) <= 4:
            credit.append(tmp)

    for i in bs.findAll("td", {"width": "9%"}):  # 上课时间
        class_time.append(i.get_text().strip())

    for i in bs.findAll("td", {"width": "8%"}):  # 教室
        if i.get_text().strip() != "课程种类":
            room.append(i.get_text().strip())
    for i in bs.findAll("td", {"width": "10%"}):  # 首次上课日期和课程类别（日期和汉字）
        if "-" in i.get_text().strip():
            first_class.append(i.get_text().strip())

    class_info = np.array([teacher, class_name, credit, class_time, room, first_class]).transpose()
    return pd.DataFrame(class_info, columns=["teacher", "class_name", "credit", "class_time", "room", "first_class"])


# 爬取所有课程信息

# 利用登陆后的cookies跳过验证码，需要每次使用前在浏览器上登录后修改此项
cookies = {"UM_distinctid": "15f5348664898f-06eb775bea99e6-102c1709-12b178-15f534866496f1",
           "JSESSIONID": "RkLJZ82bx4nhzbQHHCJf1yhB7zGZcGhnlQQpzr5xnZ8hp1J0Hs32!-1872357982",
           "BIGipServeryjs": "3372722368.53536.0000",
           "BIGipServerruc_portal": "3909593280.42271.0000",
           "renmin_university_ypy_username": "<学号>"}


session = masterlogin(cookies=cookies)
school_list = get_school_list(session=session, cookies=cookies)

result = pd.DataFrame()

# 获取本人课程
c = session.get("http://portal.ruc.edu.cn/ypy/checkCourses_main.do", cookies=cookies)
result = pd.concat([result, get_class_info(c.text)], ignore_index=True)

# 获取本人能看到的所有课程
for s in school_list:
    c = get_school_courses_html(scode=s, sname=school_list[s], session=session, cookies=cookies)
    result = pd.concat([result, get_class_info(c)], ignore_index=True)

result.to_csv("Master_Class.csv", index=False)