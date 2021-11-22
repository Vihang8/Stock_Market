# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 03:00:02 2020

@author: vihan
"""

#-------------------importing libraries-------------#
import numpy as np
import csv
import matplotlib.pyplot as plt
import random

#-----------------------opening the csv file--------------------------#
with open('C:\CSU\Spring 2020\ML and AS\Computer Assignment\HistoricalPrices.csv','r') as stock_csv:
    stock_info = list(csv.reader(stock_csv, delimiter=","))

closing_stock = []
for j in range(1,1005):
    closing_stock.append(float(stock_info[j][4]))

closing_stock_array = np.asarray(closing_stock)
x_ar_flip = closing_stock_array

x_ar = np.flipud(x_ar_flip) #flipping the array to start training from 2016
x_ar_wh = x_ar # Making a copy of the original data for Wiener Hopf solution



#-------------------Plotting the closing stock prices-------------#
plt.figure(0)
plt.plot([i for i in range(0,len(x_ar))], x_ar)

#-----------Normalization of the data-----------------#
x_ar_mean = np.mean(x_ar[0:336])
x_ar = x_ar - x_ar_mean 

x_ar_variance = np.mean((x_ar[0:336]-x_ar_mean)**2)
x_ar = x_ar/x_ar_variance

#------------Modeling noise------------#
x_ar_norm_var = np.var(x_ar)
print(x_ar_norm_var)
noise = np.random.normal(0,x_ar_norm_var,1004)

#print(x_ar)

#----------Initializing lists for plotting---------#
train = []
Valid = []
cost = []
prediction = []


#-------------------------Initializing the weight vector------------#
w = np.array([0.9,-0.1,0.1])    # Can be initialized as ([random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)])
#w_init = w         
#print(w_init)

#--------------Selecting the Hyper-paramters------------------#
mu = 10**(-2)
num_iter = 50000
stoppingCriteria = 0

#---------------------Training and Validation----------------#
for k in range(0,num_iter):
    for i in range(0,334):
        x = np.array([x_ar[i+2],x_ar[i+1],x_ar[i]])
        error_gradient =  (x_ar[i+3]-(x_ar[i+2]*w[0]+x_ar[i+1]*w[1]+x_ar[i]*w[2]+noise[i]))*x
        w = w + mu*error_gradient 
        if np.linalg.norm(error_gradient,2) == 0:
            break
        
           
    cost.append((x_ar[i+3]-(x_ar[i+2]*w[0]+x_ar[i+1]*w[1]+x_ar[i]*w[2]+noise[i]))**2)
    
    trainingerror = 0.0        
    for i in range(0,334):
        trainingerror +=  (x_ar[i+3]-(x_ar[i+2]*w[0]+x_ar[i+1]*w[1]+x_ar[i]*w[2]+noise[i]))
        trainingerror /= 334
        
    train.append(trainingerror)        
    
    #----------------------Question 5-------------------#
    validationerror = 0.0        
    for i in range(334,668):
        validationerror +=  (x_ar[i+3]-(x_ar[i+2]*w[0]+x_ar[i+1]*w[1]+x_ar[i]*w[2]+noise[i]))
        validationerror /= 334
    Valid.append(validationerror)        
    
    if k>=1:
        if cost[-2]<cost[-1]:
            stoppingCriteria+=1
            if stoppingCriteria==1:
                break
        else:
            stoppingCriteria=0

print("\nFinal value of training error "+str(trainingerror))
plt.figure(2)
plt.plot([i for i in range(0,len(train))], train)
plt.title('Training error')
plt.xlabel('iteration')
plt.ylabel('Training Error')

print("\nFinal value of validation error "+str(validationerror))
plt.figure(3)
plt.plot([i for i in range(0,len(Valid))], Valid)
plt.title('Validation error')
plt.xlabel('iteration')
plt.ylabel('Validation Error')




#---------------Learning Curve-------------------------#
plt.figure(1)
plt.plot([i for i in range(0,len(cost))], cost)
plt.title('Learning Curve')
plt.xlabel('iteration')
plt.ylabel('Cost')

#--------------Wiener-Hopf Solution------------------#

X = np.array([list(x_ar_wh[2:-1]),list(x_ar_wh[1:-2]),list(x_ar_wh[0:-3])]) 
Xtranspose = np.transpose(X)

Rxx = np.dot(X,Xtranspose)

target_vector = np.array([list(x_ar_wh[3:])])

Rxd = np.dot(X,np.transpose(target_vector))

w_star = np.dot(np.linalg.inv(Rxx),Rxd)

print("\nWiener-Hopf Solution")
print(w_star)    
print("\nLMS solution")
print(w)

#-----------------Convergence behavior and accuracy------#

X_trace = np.array([list(x_ar[2:-1]),list(x_ar[1:-2]),list(x_ar[0:-3])]) 
Xtranspose_trace = np.transpose(X_trace)

Rxx_trace = np.dot(X_trace,Xtranspose_trace)

misadjustment = mu*Rxx_trace.trace()/(1-mu*Rxx_trace.trace())
print("\nMisadjustment")
print(misadjustment)

#----------------Question 6 (Testing Error)----------------#

testingerror = 0.0        
for i in range(668,1001):
    testingerror +=  (x_ar[i+3]-(x_ar[i+2]*w[0]+x_ar[i+1]*w[1]+x_ar[i]*w[2]))
    testingerror /= 334
print("\n Final value of Testing error")
print(testingerror)

for i in range(668,1002):
    prediction.append((x_ar[i+2]*x_ar_variance+x_ar_mean)*w_star[0]+(x_ar[i+1]*x_ar_variance+x_ar_mean)*w_star[1]+(x_ar[i]*x_ar_variance+x_ar_mean)*w_star[2])
    
plt.figure(0)
plt.plot([(668+i) for i in range(0,len(prediction))], prediction)
