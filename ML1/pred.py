from sklearn.linear_model import LinearRegression
import numpy as np
x= np.array([[1], [2], [3], [4],[5],[6],[7]])
y=np.array([30,50,70,90,100,120,140])
model=LinearRegression() #The dataype of x is converted internally to a numpy array
model.fit(x,y)
print(model.predict([[5]]))
print(type(x))