import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def knn(samples, labels, query, k, distance_function=euclidean_distance):

  print('Calculando distâncias com a função {}'.format(highlight(distance_function.__name__)))

  # Salva as distancias e suas classes em um array E
  E = []
  for index, row in samples.iterrows():
    distance = distance_function(row.values, query)
    pair = (distance, labels[index])
    E.append(pair)

  print('{} das distâncias calculadas e suas classes'.format(highlight(3)))
  print_distances_and_classes(E[:3])
  done()

  # Ordena as distancias obtidas anteriormente
  print('Ordenando valores por distância')
  sorted_E = sorted(E, key=lambda pair: pair[0])
  done()

  # Pega somente as k distancias mais próximas
  print('As {} distâncias mais próximas e suas classes:'.format(highlight(k)))
  distances_and_classes = sorted_E[0:k]
  print_distances_and_classes(distances_and_classes)
  done()

  # Conta quantas vezes cada classe aparece
  print('Contagem de ocorrências de cada classe')
  classes = [i[1] for i in distances_and_classes]
  votes = Counter(classes)
  print_votes(votes)

  # Obtêm a classe mais comum
  max_class = max(votes, key=votes.get)
  done()

  # Retorna a classe
  return max_class


def save(title):
  plt.savefig(f'./img/knn/{title}.png')
  plt.clf()


def correlation(df):
  print('Plotando matrix de correlação')

  matrix = df.corr()

  plt.figure()
  sns.heatmap(matrix, cmap='Greens')
  save('correlation')

  done()


def pairplot(df, labels):
  print('Plotando pairplot')

  plt.figure()
  sns.pairplot(df, hue='label', corner=True)
  save('pairplot')

  done()


def rgb(r, g, b):
  return (r/255.0, g/255.0, b/255.0)


def append_query(df, query, result):
  is_query = 1
  new_row = {
    'sepal_length': query[0],
    'sepal_width': query[1],
    'petal_length': query[2],
    'petal_width': query[3],
    'label': result,
    'is_query': is_query
  }

  return df.append(new_row, ignore_index=True)


def scatterplot(df, query, result):
  print('Plotando scatterplot')

  df_with_query = append_query(df, query, result)

  plt.figure()
  sns.scatterplot(
    data=df_with_query,
    x="petal_width",
    y="petal_length",
    hue="label",
    style='is_query',
    size='is_query',
    sizes=(50, 51)
  )
  save('scatterplot')

  done()


if __name__ == "__main__":
  sns.set()

  header('Calculando classificação com o algoritmo K-NN')

  df, labels = load_dataframe('./data/iris.csv')
  print(df, end='\n\n')

  query = [7.3, 3.5, 4.9, 1.1]
  k = 11

  classification = knn(df, labels, query, k)

  classification = knn(df, labels, query, k, distance_function=cosine_similarity)

  title('Classificado como {}'.format(classification))
  print('A entrada {} foi classificada como {}\n'.format(highlight(query), highlight(classification)))

  df['label'] = labels

  pairplot(df, labels)
  correlation(df)

  df['is_query']= np.zeros(len(df))
  scatterplot(df, query, classification)
