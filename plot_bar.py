import matplotlib.pyplot as plt
import csv
import sys
import numpy as np

c0 = []
c1 = []
c2 = []
headers = None
with open(sys.argv[1],'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if not headers:
            headers = row
            continue
        print(row)
        c0.append(row[0].replace('conv','').replace('_weight','').replace('downsample','dnsmp'))
        c1.append(float(row[1]))
        c2.append(float(row[2]))

x = np.arange(len(c0))        
#print(headers)
fig, ax = plt.subplots()
#ax = fig.add_axes([0,0,1,1])
ax.bar(x, c1)
ax.bar(x+0.2, c2, width=0.3)
ax.set_xticks(x, c0)
ax.set_xticklabels(c0)
plt.xticks(x, c0, rotation=90)
ax.set_ylabel('Sparsity')
ax.set_title(sys.argv[2])
ax.set_yticks(np.arange(0, 1, 0.2))
#plt.xlim(0,200)
plt.legend(headers[1:5])
fig.set_size_inches(10, 10)
fig.show()
print('writing to',sys.argv[1]+'.pdf')
fig.savefig(sys.argv[1]+'.pdf')
