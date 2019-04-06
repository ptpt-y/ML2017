#################################################
#    >File Name: gd.py
#    >Author: Tingyu Peng
#    >mail: PengTingyu.d@gmail.com
#    >Created Time: 2019年04月03日 星期三 14时39分15秒
#################################################
#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import pandas
import csv, os
import matplotlib.pyplot as plt
import random
import math
import sys

# get path
inputTrain = sys.argv[1]
inputTest = sys.argv[2]
output = sys.argv[3]

def ada(X, Y, w, eta, iteration, lambdaL2):
    s_grad = np.zeros(len(X[0]))
    cost_history = []
    for i in range(iteration):
        hypo = np.dot(X,w)
        loss = hypo - Y
        cost = np.sum(loss**2)/(2*len(X))
        cost_history.append(cost)
 
        grad = np.dot(X.T,loss)/len(X)+lambdaL2*w
        s_grad += grad**2
        ada = np.sqrt(s_grad)
        w = w - eta*grad/ada
    return w, cost_history

# in data, each row/dimension representes a kind of pollutino info
data = []
# print(np.array(data).shape) # (0,)
for i in range(18):
    data.append([])
# print(np.array(data).shape) # (18,0)

# read data from train.csv
n_row = 0
with open(inputTrain,"r",encoding='big5') as text:
    row = csv.reader(text, delimiter=',')
    for r in row:
        if n_row != 0:
            for i in range(3,27):
                if r[i] != "NR":
                    data[(n_row-1)%18].append(float(r[i]))
                else:
                    data[(n_row-1)%18].append(float(0))
        n_row = n_row + 1
# print(np.array(data)) # result: show_nd.array_data.png
# print(np.array(data).shape) # (18,5760) 12*20*24=12*480=5760


# parse data to trainX and trainY
x = []
y = []
for i in range(12):
    for j in range(471):
        x.append([])
        for t in range(18):
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s])
        y.append(data[9][480*i+j+9])

trainX = np.array(x) # each row has 9*18 numbers, every 9 numbers represent a pollution
trainY = np.array(y)
# print(trainX.shape) # (5652,9*18) 12*(480-9) hours are not limited by day
# print(trainY.shape) # (5652,)

# parse test data
test_x = []
n_row = 0
with open(inputTest,"r") as text:
    row = csv.reader(text,delimiter=",")
    for r in row:
        if n_row%18==0:
            test_x.append([])
            for i in range(2,11):
                test_x[n_row//18].append(float(r[i]))
        else:
             for i in range(2,11):
                 if r[i]!="NR":
                    test_x[n_row//18].append(float(r[i]))
                 else:
                    test_x[n_row//18].append(float(0))
        n_row = n_row+1
test_x = np.array(test_x)

# parse anser
ans_y = []
n_row = 0
with open('data/ans.csv',"r") as text:
    row = csv.reader(text,delimiter=",")
    for r in row:
        ans_y.append(r[1])
    ans_y = ans_y[1:]
    ans_y = np.array(list(map(int,ans_y)))

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
trainX = np.concatenate((np.ones((trainX.shape[0],1)),trainX), axis=1)

# init weights
w = np.random.randn(trainX.shape[1])*0.01
# training
w_ada, cost_his_gd = ada(trainX, trainY, w, eta=1, iteration=20000, lambdaL2=0)
# testing 
y_ada = np.dot(test_x, w_ada)
# normal equation
w_ne = np.linalg.inv(trainX.T.dot(trainX)).dot(trainX.T).dot(trainY)
y_ne = np.dot(test_x,w_ne)

# csv format
ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w_ada,test_x[i])
    ans[i].append(a)

filename = output
with open(filename,"w+") as text:
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["id","value"])
    for i in range(len(ans)):
        s.writerow(ans[i])
# plot cost_his_gd
plt.plot(np.arange(len(cost_his_gd[3:])),cost_his_gd[3:],'g',label='ada')
plt.title('Training Process')
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__),"figures/TrainingProcess"))
plt.show()
# plot result
plt.figure()
plt.subplot(131)
plt.title('normal equation')
plt.xlabel('dataset')
plt.ylabel('pm2.5')
plt.plot(np.arange((len(ans_y))),ans_y,"r")
plt.plot(np.arange(240),y_ne,"b")
plt.subplot(132)
plt.title('ada')
plt.xlabel('dataset')
plt.ylabel('pm2.5')
plt.plot(np.arange((len(ans_y))),ans_y,"r")
plt.plot(np.arange(240),y_ada,"g")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__),"figures/Compare"))
plt.show()
        

