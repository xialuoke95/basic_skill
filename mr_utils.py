# -*- coding: utf-8 -*-

import os, pdb, sys, logging, signal, argparse, time, json
import subprocess, gc
from datetime import datetime, timedelta

HADOOP_HOME="/opt/tiger/yarn_deploy/hadoop-2.6.0-cdh5.4.4"
HADOOP_BIN = "%s/bin/hadoop" % HADOOP_HOME
STREAMING="/opt/tiger/yarn_deploy/hadoop-2.6.0-cdh5.4.4/share/hadoop/tools/lib/hadoop-streaming-2.6.0-cdh5.4.4.jar"


def check_success(path):
    cmd = "%s fs -test -e %s/_SUCCESS" %(HADOOP_BIN, path)
    logging.info("exec cmd [%s]", cmd)
    ret = os.system(cmd)
    if ret == 0:
        return True
    return False


def get_dir_files(path):
    l = []
    try:
        cmd = "%s fs -ls %s" %(HADOOP_BIN, path)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        output, err = p.communicate()
        for row in output.split('\n'):
            pp = row.split(' ')[-1]
            l.append(pp)
    except:
        pass
    return l


def run_cmd(cmd, msg=None):
    logging.info("exec: [%s]"%cmd);
    ret = os.system(cmd)
    if ret != 0:
        logging.error("%s: [%s]"%(msg, cmd))
    return ret

def check_path(path):
    cmd = "%s fs -test -e %s" %(HADOOP_BIN, path)
    logging.info("exec cmd [%s]", cmd)
    ret = os.system(cmd)
    if ret == 0:
        return True
    return False


def upload_file(lpath, hpath):
    cmd = "%s fs -copyFromLocal -f %s %s" %(HADOOP_BIN, lpath, hpath)
    logging.info("exec cmd [%s]", cmd)
    ret = os.system(cmd)
    if ret == 0:
        return True
    return False


def download_file(path1, path2):
    cmd = "%s fs -copyToLocal %s %s" %(HADOOP_BIN, path1, path2)
    logging.info("exec cmd [%s]", cmd)
    ret = os.system(cmd)
    if ret == 0:
        return True
    return False

def get_hdfs_file(hdfs_dir, file_format, local_file):
    hdfs_dirname = hdfs_dir
    current = datetime.now()
    dtr = current.strftime('%Y%m%d%H')
    hdfs_path = '%s.%s' % (file_format, dtr)
    path1 = hdfs_dirname + hdfs_path
    download_file(path1, local_file)
    if not os.path.exists(local_file):
        if current.minute <= 20:
            current = current - timedelta(hours=1)
            dtr = current.strftime('%Y%m%d%H')
            local_path = '%s.%s' % (file_format, dtr)
            path1 = hdfs_dirname + local_path
            download_file(path1, local_file)
        else:
            raise Exception('%s not found' % local_file)
    return local_file

from pyutil.hiveserver2 import connect
def execute_hive_query(sql = "", job_name = "nlp_mr_util", user = "tiger", large_memory=False):
    logging.info('start to query hive: %s', sql)
    start = datetime.now()
    result = []
    with connect('hive', cluster='haruna_noauth', username=user) as client:
        with client.cursor() as cursor:
            configuration = {
                'mapreduce.job.queuename': 'offline.data',
                'mapreduce.job.name': '%s_%s' % (job_name, user),
                'mapreduce.job.priority': 'HIGH',

            }
            if large_memory:
                configuration.update({
                    'mapreduce.map.java.opts': '-Xmx10680m',
                    'mapreduce.reduce.java.opts': '-Xmx10680m',
                    'mapreduce.map.memory.mb': '16384',
                    'mapreduce.reduce.memory.mb': '16384',
                })
            cursor.execute(sql, configuration=configuration, async=True)
            while not cursor.is_finished():
                try:
                    print cursor.get_log(start_over=False)
                except:
                    pass
            if cursor.is_success():
                result = cursor.fetchall()
    time_delta = (datetime.now() - start).total_seconds() / 60
    logging.info('[%s]get data from hive, len: %s, cost = %s minutes', job_name, len(result), time_delta)
    return result

if __name__ == '__main__':
    print 'ok'
    lpath = 'data/doc_score.2016122901'
    #hpath = '/recommend/data/nipeng/sentiment_scorer/sentiment2/'
    #print upload_file(lpath, hpath)
    sql = """
        select
            group_id,
            impr_feed,
            impr_channel,
            impr_new_user,
            impression
        from
            push.low_quality_group_impression_daily
        where
            date = '20180419'
            and impression > 100
    """
    print sql
    tmp = execute_hive_query(sql = sql, job_name="get_impression", user = "wangyuehan")
    print tmp

