import matplotlib.pyplot as plt
import csv
import sys
import numpy as np
import re
import random
import matplotlib.colors as mcolors
from matplotlib.pyplot import figure

lrs = []
vals = []
legs = []
headers = None


for i in range(1,len(sys.argv)):
    with open(sys.argv[i],'r') as csvfile:
        log_leg = re.sub(r'@.*','',re.sub(r'.*net18@','' ,sys.argv[i])).replace('_True_fixedStep','F').replace('_True_specified','S').replace('_True_plateau', 'P').replace('_False_fixedStep','base').replace('_32_','').replace('000000','M').replace('_1_0.9','').replace('100','').replace('00000',',M') 
        plots = csv.reader(csvfile, delimiter='\t')
        log_fin_acc= re.sub(r'_.*','', sys.argv[i])
        log_lr=[]
        log_val = []
        for row in plots:
#            print(row)
            if not headers:
                headers = row
                continue
            log_lr.append(float(row[0]))
            log_val.append(float(row[4]))
        legs.append(headers[4]) #'('+log_fin_acc+'%)' + log_leg)
        lrs.append(log_lr)
        vals.append(log_val)
    headers = None


fig=figure(num=None, figsize=(3.33, 1.6), dpi=80, facecolor='w', edgecolor='k')
SMALL_SIZE=8
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#print(len(vals))
#print(headers)
cmap = plt.cm.get_cmap('hsv', len(sys.argv))
#for i in range(1,2):
#    plt.plot(np.transpose(vals[i-1]), c=mcolors.to_rgb('black'), linewidth=1)
#for i in range(2,len(sys.argv)):
#    plt.plot(np.transpose(vals[i-1]), linewidth=.6, c=cmap(i))  #c=(random.random(), random.random(), random.random())) #c=cmap(i))

i=1
plt.plot(np.transpose(vals[i-1]), c=mcolors.to_rgb('black'), linewidth=1)
i=2
plt.plot(np.transpose(vals[i-1]), linewidth=.6, c=mcolors.to_rgb('blue'), linestyle='-.')
i=3
plt.plot(np.transpose(vals[i-1]), linewidth=.6, c=mcolors.to_rgb('purple'), linestyle='--')
i=4
plt.plot(np.transpose(vals[i-1]), linewidth=.6, c=mcolors.to_rgb('red'))

plt.ylim(0, 80)
plt.xlim(0,100)
plt.ylabel('Validation Accuracy')
plt.xlabel('Epochs')
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.legend(legs, loc='lower right', fontsize=8)
plt.grid(True, which='both')

fig.savefig('test.pdf',bbox_inches='tight')
