import pandas as pd
import datetime


objects = 3 #number of colours used
iterations=60 #number of times all colours are shown
srate = 256 #sampling rate
timeint = 4 #duration of one trial
timepts = timeint * srate
df = pd.read_csv('/Users/mahima/research/museappdata/mahimafinal3.csv', index_col=None)
df1=pd.read_csv('/Users/mahima/research/museappdata/mahimafinal3.csv')
indexNames = df[df['Elements'] == '/muse/elements/jaw_clench'].index

colours=[None] * (iterations)
i=0
m=""
firstind = indexNames[0]
with open('/Users/mahima/research/museappdata/logs/mahimafinal3_museexpfinal_2020_Apr_29_0201.log') as f:
    for line in f:
        
        if "blue.png')])" in line:

            colours[i]='blue'
            i=i+1
        if  "red.png')])" in line:

            colours[i]='red'
            i=i+1
        if  "green.jpg')])" in line:

            colours[i]="green"
            i=i+1
            


firsttime = df['TimeStamp'].iloc[firstind]
indexfinal=[None] * (iterations+1)
findtime = [None] * (iterations)
indexfinal[0]=firstind
for i in range(0,iterations):

    findtime[i] = datetime.datetime.strptime(firsttime, "%Y-%m-%d %H:%M:%S.%f") + datetime.timedelta(0,
                                                                                                     (i+1)  * timeint)
    

index = [None] * (iterations) #stores the required timestamps 


df['TimeStamp']= pd.to_datetime(df['TimeStamp'])
df.set_index(['TimeStamp'], inplace=True)
df = df.loc[~df.index.duplicated(keep='first')]
df = df.sort_index()


for i in range(0, iterations):

    index[i] = str(df.index[df.index.get_loc(findtime[i].strftime("%Y-%m-%d %H:%M:%S.%f"), method='nearest')])
   


for i in range(iterations):

    index[i]=datetime.datetime.strptime(index[i], "%Y-%m-%d %H:%M:%S.%f")
    index[i]=index[i].strftime("%Y-%m-%d %H:%M:%S.%f")
    index[i]=index[i][:-3]
    #index[i] = datetime.datetime.strptime(index[i], "%Y-%m-%d %H:%M:%S.%f")

'''
for i in range (0,objects):
    index[i]=datetime.datetime.fromtimestamp(index[i])

'''
#df1['TimeStamp']= pd.to_datetime(df1['TimeStamp'])


for i in range (0, iterations):

    final = df1[df1['TimeStamp'] == index[i]].index
    indexfinal[i+1]=final[0]
    

#indexfinal[iterations]=indexNames[1]
#colour=['red','green','blue','blue','red','green','red','green','blue','green','red','blue','blue','green','red']
number=['0','0','0','1','1','1','2','2','2','3','3','3','4','4','4','5','5','5','6','6','6','7','7','7','8','8','8','9','9','9','10','10','10','11','11','11','12','12','12','13','13','13','14','14','14','15','15','15','16','16','16','17','17','17','18','18','18','19','19','19']

for i in range (0,iterations):
    d=df1.iloc[indexfinal[i]:]
    
    d=d.iloc[:(indexfinal[i+1]-indexfinal[i])]
    
  

    d.to_csv (r'/Users/mahima/research/df1'+colours[i]+number[i]+'.csv', index = False, header=True)
print(colours)