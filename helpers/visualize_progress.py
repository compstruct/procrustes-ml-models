import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

table = pd.read_csv("tuning/progress.csv")
table.head()
run_cnt=len(table)

print(table['Epochs']) 
print(table['Runs'])
print(np.arange(1,run_cnt))
#print(np.arange(1,run_cnt).reduce())

plt.barh(np.arange(run_cnt), table['Epochs'])
plt.title("Training Progress")
#plt.set_yticklabels(table['Runs'])

plt.xlabel("runs")
plt.ylabel("epochs")
#plt.show()
plt.savefig('tuning/progress.png')


