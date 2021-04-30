import pandas as pd
import numpy as np
from math import sqrt
from collections import Counter
from style import *
from utils import load_dataframe


def euclidean_distance(x, y):
  total = 0
  for index in range(len(x)):
    total = total + ((x[index] - y[index]) ** 2)

  return sqrt(total)


def manhattan_distance(x, y):
  total = 0
  for index in range(len(x)):
    total = total + abs((x[index] - y[index]))

  return total


def cosine_similarity(x, y):
  product = np.dot(x,y)
  x_norm = np.linalg.norm(x)
  y_norm = np.linalg.norm(y)

  return 1 - (product / (x_norm * y_norm))


def knn(samples, labels, query, k, distance_function):
  E = []

  print('Calculando distâncias com a função {}'.format(highlight(distance_function.__name__)))
  for index, row in samples.iterrows():
    distance = distance_function(row.values, query)
    E.append((distance, labels[index]))
  done()

  print('Ordenando valores por distância')
  sorted_E = sorted(E, key=lambda pair: pair[0])
  done()

  print('As {} distâncias mais próximas e suas classes:'.format(highlight(k)))
  distances_and_classes = sorted_E[0:k]
  print_distances_and_classes(distances_and_classes)
  done()

  print('Contagem de ocorrências de cada classe')
  classes = [i[1] for i in distances_and_classes]
  votes = Counter(classes)
  print_votes(votes)

  max_class = max(votes, key=votes.get)
  done()

  title('Classe encontrada!')

  return max_class


if __name__ == "__main__":
  header('Calculando classificação com o algoritmo K-NN')

  df, labels = load_dataframe('./data/iris.csv')
  done()
  print(df, end='\n\n')

  query = [7.3, 3.5, 4.9, 1.1]
  k = 11
  classification = knn(df, labels, query, k, euclidean_distance)

  print('A entrada {} foi classificada como {}'.format(highlight(query), highlight(classification)))
