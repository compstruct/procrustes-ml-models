import matplotlib.pyplot as plt
import csv
import sys
import numpy as np

c0 = []
c1 = []
c2 = []
c3 = []
c4 = []
headers = None
with open(sys.argv[1],'r') as csvfile:
    plots = csv.reader(csvfile, delimiter='\t')
    for row in plots:
        if not headers:
            headers = row
            continue
        c0.append(float(row[0]))
        c1.append(float(row[1]))
        c2.append(float(row[2]))
        c3.append(float(row[3]))
        c4.append(float(row[4]))
#print(headers)
plt.plot(np.transpose([c1,c2,c3,c4]))
plt.ylim(0, 75)
plt.xlim(0,200)
plt.legend(headers[1:5])
plt.savefig(sys.argv[1]+'.pdf')
