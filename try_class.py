# -*- coding: utf-8 -*-

class Father(object):
    def __init__(self, s, time):
        self.s = s
        self.time = time

    def run(self):
        self.a = 1

class Children(Father):
    def __init__(self,**params):
        # s = params['s']
        # time = params['time']
        # super(Children, self).__init__(**params['s'],**params['time'])
        super(Children, self).__init__(params['s'],params['time'])
        # super(Children, self).__init__(s,time)

    def run(self, b):
        super(Children, self).run()
        self.b = b

class Childrens(Children):
    def __init__(self,**params):
        # s = params['s']
        # time = params['time']
        # super(Children, self).__init__(**params['s'],**params['time'])
        super(Childrens, self).__init__(**params)
        super(Childrens, self).__init__(s = params['s'], time = params['time'])
        # super(Children, self).__init__(s,time)

    def run(self, b):
        super(Childrens, self).run()

AA = Children(**{'s':'a','time':'b'})
AA = Children(s = 'a', time = 'b')
AA.run(3)
print AA.b

BB = Childrens(s = 'a', time = 'b')

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

a = Ass()
a = Ass(b = 4)
