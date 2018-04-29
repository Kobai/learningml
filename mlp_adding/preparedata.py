import csv
import random
import numpy as np

# Basically just an RNG to get numbers and sums


final_data = [['num1','num2','numsum']]
myData = []

num1= []
for i in range (10000):
    num1.append(random.randrange(1000))
myData.append(num1)

num2 =[]
for i in range(10000):
    num2.append(random.randrange(1000))
myData.append(num2)

numsum=[]
for i in range(10000):
    numsum.append(num1[i] + num2[i])
myData.append(numsum) 

myData = np.transpose(myData)

myFile = open('training.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(final_data)
    writer.writerows(myData)