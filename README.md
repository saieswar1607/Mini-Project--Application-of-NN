# Mini-Project--Application-of-NN

My project can helps you to analyze and makes you to understand in better manner. this type of application may helps in various NN models.The main usage of this model is that you can apply to any type of Neural network model. Agenda is to implement the Simple NN model.

## Project Title: Building own Neural Network Model

## Project Description :

This Neural Network Model helps to analyze how a Neural Network propagates in forward direction and backward direction and also i have mentioned how to utilize the non linear(activation function).The important thing is that i tested the model with some random values and i checked the performance of the model ,the model was nearly 96% accurate.

## Algorithm:
```
1.Import the require modules
2.I Created my own dataframe with random .rand
3.I plot the Inline map to visualize the points
4.Assign the random weights and Bias
5.Feed some data to the forward propagation
6.Then calculate the Gradient Descent (by differentiation)
7.Test the model with the random set of points
```
## Program:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
X=np.vstack([(np.random.rand(10,2)*5),(np.random.rand(10,2)*10)])
Y=np.hstack([[0]*10,[1]*10])
dataset=pd.DataFrame(X,columns={"X1","X2"})
dataset["Y"]=Y
dataset
plt.plot(dataset,label='Inline label')
plt.legend(["X2","X1","Y"])
Z=np.zeros((20,2))
for i in range(20):
    Z[i,Y[i]]=1
X.shape
wi_1=np.random.randn(3,2)
Bi_1=np.random.randn(3)
wi_2=np.random.randn(3,2)
Bi_2=np.random.randn(2)
wi_1
Bi_2
wi_2
wi_1.T
X
X.shape
X.dot(wi_1.T)
def forward_prop(X,wi_1,Bi_1,wi_2,Bi_2):
    M=1/(1+np.exp(-(X.dot(wi_1.T)+Bi_1)))
    A=M.dot(wi_2)+Bi_2
    expA=np.exp(A)
    Y=expA/expA.sum(axis=1,keepdims=True)
    return Y,M
forward_prop(X,wi_1,Bi_1,wi_2,Bi_2)
def diff_wi_2(H,Z,Y):
    return H.T.dot(Z-Y)
def diff_wi_1(X,H,Z,output,wi_2):
    dz=(Z-output).dot(wi_2.T)*H*(1-H)
    return X.T.dot(dz)
def diff_B2(Z,Y):
    return (Z-Y).sum(axis=0)
def diff_B1(Z,Y,wi_2,H):
    return((Z-Y).dot(wi_2.T)*H*(1-H)).sum(axis=0)
learning_rate=1e-3
for epoch in range(5000):
    output,hidden=forward_prop(X,wi_1,Bi_1,wi_2,Bi_2)
    wi_2+=learning_rate*diff_wi_2(hidden, Z,output)
    Bi_2+=learning_rate*diff_B2(Z ,output)
    wi_1+=learning_rate*diff_wi_1(X,hidden,Z,output,wi_2).T
    Bi_1+=learning_rate*diff_B1(Z ,output,wi_2,hidden)
x_test=np.array([9,7])
hidden_output=1/(1+np.exp(- x_test.dot(wi_1.T)-Bi_1))
Outer_layer_output=hidden_output.dot(wi_2)+Bi_2
expA=np.exp(Outer_layer_output)
Y=expA/expA.sum()
print(" prob of class 0>>>>>>> {} \n prop of class 1>>>>> {}".format(Y[0],Y[1]))
```
## Output:

![image](https://user-images.githubusercontent.com/93427522/205952380-1794bbe5-fbdd-4d2c-89df-59972c86d455.png)

![image](https://user-images.githubusercontent.com/93427522/205952462-816d3bf9-0c0d-4bf5-8941-01d05bc5a1ac.png)


## Advantage :
```
1.To Understand the process in practical manner.
2.This type of Model works as blue print for another projects (like templet)
3.Every one can understand easily.
```
## Result:

Thus the implementation of simple Neural Network Model via forward propagation and backward propagation was executed successfully ,by our own data points and dataset.
