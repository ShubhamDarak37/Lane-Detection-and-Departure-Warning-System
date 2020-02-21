import pandas as pd
Data=pd.read_csv("Tips.csv")
df1=pd.DataFrame(Data, columns= ['Time', 'TotalBill', 'Tips'] )
df1
 df1=Data[ ['Time', 'TotalBill', 'Tips'] ]
 df1=Data.iloc[:,0:2]
 df1=Data.loc[:, ['Time', 'TotalBill', 'Tips'] ]
Data1=pd.read_excel("Tips1.xlsx")
 Data2 = pd.DataFrame.join(Data, Data1, on=None, how='left')
 Data2 = pd.DataFrame.join(Data, Data1, on=None, how='left')
 Data2 = pd.concat(Data, Data1, join='outer')
 Data2 = pd.merge(Data, Data1, how='left')
Data2

Data3 = Data2.copy()
Data3

 Data3.groupby('Day')[['Tips']].aggregate(sum)
 Data3.groupby('Gender')['Time'].aggregate(sum)
 Data3.groupby(['Gender', 'Time'])['Time'].count().unstack()
 pd.crosstab(index = Data3['Gender'], columns = Data3['Time'], normalize = False)
 Data3.pivot_table('Time', index='Gender', columns=Data3.Time.values, aggfunc=len)
Data3.hist(column='TotalBill')
TotalBill
Data3.plot(kind='barh')

Data3['Day'].value_counts().plot(kind='bar');

 Data3.groupby('Day').aggregate('sum')
 Data3['Tips'].mean()
 Data3.groupby('Day').apply(lambda x: x.mean())
 Data3.groupby('Day').apply(mean)

import copy
x = [5,4,3,2,1]
y=[7,8,9]
z=[x,y]
a=copy.deepcopy(z)
b=copy.copy(z)
x[2]=6
print('a=',a,'b=',b)