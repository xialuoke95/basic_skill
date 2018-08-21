# -*- coding: utf-8 -*-

import csv
import pandas as pd
with open('try.csv','wb') as final_data:
    writer = csv.writer(final_data)
    writer.writerow([1,2,3])

with open('try.csv','rb') as final_data:
    reader = csv.reader(final_data)
    row = reader.next()
    print row
    print type(row[1])

a = pd.read_csv('try.csv',header = None)
print type(a.iloc[0,0])


