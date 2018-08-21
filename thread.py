# -*- coding: utf-8 -*-

import threading
import time

num = 0
mutex = threading.Lock()

def run(n):
    print('current task', n)

class MyThread(threading.Thread):
    def __init__(self):
        super(MyThread, self).__init__()
        # self.n = n

    # def run(self):
    #    print('current task:', self.n)

    def run(self):
        global num
        time.sleep(1)

        # 超时判断是否得到锁
        if mutex.acquire(1):
            num += 1
            msg = self.name + 'num is' + str(num)
            print msg
            mutex.release()

def count(n):
    while n > 0:
        n -= 1

## t1未必执行完 可能就切了

if __name__ == "__main__":
    t1 = threading.Thread(target=run, args=('thread 1',))
    t2 = threading.Thread(target=run, args=('thread 2',))

    # t1 = MyThread('the 1')
    # t2 = MyThread('the 2')

    t1.start()
    t2.start()

    for i in range(5):
        t = MyThread()
        t.start()
