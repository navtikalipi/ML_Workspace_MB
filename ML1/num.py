import numpy as  np

list1=np.array([10,20,30,40])
list2=np.array([1,10,20,100])

res=list1+list2
print(res)

#find if value exists in an array- use in
#find index, use where(arrayname==num)
print(20 in list1)

#create a 2D array
l1=np.array([[1,2,3,4],
            [5,6,7,8],
            [9,10,11,12]])

print(l1.resize(4,3))
#resize=change the array structure
#reshape=does not change the original array. Assign it to view the changes