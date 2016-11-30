import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.DataFrame()

with open('/home/zach/repos/facenet/src/test_results.txt','r') as input_data:
    z = eval(input_data.readlines()[0])
    df = pd.DataFrame.from_dict([a for a in z])

em = df.drop('logits', 1)
log = df.drop('embits', 1)
print len(df)
print em.corr()
print log.corr()

chance =  (1./7)
print [round(x,2) for x in [0-chance, chance/2, chance, chance*2]]

matrix = np.zeros([7, 7],dtype=int)
for i, v in enumerate(df.logits):
    matrix[v, df.target[i]] += 1
    print 
print matrix.max() 
sns.set(style="white")
with sns.color_palette("husl", 8):
    ax = plt.axes()
    sns.heatmap(matrix,cmap="YlGnBu",ax=ax) 
    ax.invert_yaxis()
    font = {'size':'18'}
    ax.set_xlabel('Prediction', **font)
    ax.set_ylabel('Target', **font)

    
plt.show()
     