
# coding: utf-8

# In[184]:


def  compute_error(m,b,coordinates):
    totalerror=0
    for i in range(0,len(coordinates)):
        x=coordinates[i][0]
        y=coordinates[i][1]
        totalerror+=(y-(x*m+b))**2
    return totalerror / float(len(coordinates))   

s=compute_error(1,2,[[2,4],[3,6],[41,22]])
s

