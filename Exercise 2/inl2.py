####################################
# Arvid Edenheim
# 106502907
# Introduction to Data Science
####################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

k = int(input("Number of k: "))

colArr = ['b.', 'r.', 'g.', 'y.', 'k.' ]

data_csv = pd.read_csv('dataset/iris.data.txt',
    names=["Col1", "Col2", "Col3", "Col4", "Col5"])

centroids = []
counter_list = []

samples = data_csv.sample(k)

for i in samples.index:
  centroids.append([samples.Col3[i], samples.Col4[i]])

def return_distance(x1, y1, x2, y2):
  return  ((x2 - x1) ** 2) + ((y2 - y1) ** 2)

def converged(old, new):
  for i in range(0,k):
    if new[i][0] != old[i][0] or new[i][1] != old[i][1]:
      return False
  return True

while(True):
  sorted_list = []
  for input_data in range(0,len(data_csv.index)):
    closest_distance = return_distance(data_csv.Col3[input_data], data_csv.Col4[input_data], centroids[0][0], centroids[0][1])
    closest_centroid = 0
    for centroid_data in range(0,k):
      if return_distance(data_csv.Col3[input_data], data_csv.Col4[input_data], centroids[centroid_data][0], centroids[centroid_data][1]) < closest_distance:
        closest_centroid = centroid_data
        closest_distance = return_distance(data_csv.Col3[input_data], data_csv.Col4[input_data], centroids[centroid_data][0], centroids[centroid_data][1])
    sorted_list.append([closest_centroid, data_csv.Col3[input_data], data_csv.Col4[input_data]])

  counter_list = []
  
  for i in range(k):
    counter_list.append([0, 0, 0])

  for element in sorted_list:
    counter_list[element[0]][0] += 1
    counter_list[element[0]][1] += element[1]
    counter_list[element[0]][2] += element[2]

    old_centroids = centroids
    centroids = []

  for i in range(0,k):
    if counter_list[i][0] != 0:
      centroids.append([counter_list[i][1] / counter_list[i][0], counter_list[i][2] / counter_list[i][0]])

  if converged(old_centroids, centroids):
    break

for element in sorted_list:
  plt.plot(element[1], element[2], colArr[element[0] % len(colArr)], linewidth = 0.5, markersize = 5)

for i in range(0,k):
  plt.plot(centroids[i][0], centroids[i][1], 'k.', linewidth = 0.5, markersize = 10)

plt.ylabel('Column 1')
plt.xlabel('Column 2')
plt.show()