# -*- coding: utf-8 -*-

from datetime import datetime

# default is 00:00:00 for the day
datetime.strptime('20180805','%Y%m%d')

timestamp

time.time()
datetime.fromtimestamp(timestamp)

time.mktime( datetime.strptime('20180805','%Y%m%d').timetuple() )
