import pandas as pd
from scipy import stats
from numpy import savetxt
import numpy as np

arr = np.arange(1, 87, 1)
df=pd.read_csv("test.csv")
df=stats.zscore(df)
df = pd.DataFrame(data=df, columns=arr)

df.to_csv(r'test.csv', index = False)


