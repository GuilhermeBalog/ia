import pandas as pd
import numpy as np
from utils import load_dataframe
from style import *


DELTA = 0.1


def binary_degree(net, delta):
    if net >= delta:
        return 1
    else:
        return 0


def bipolar_degree(net, delta):
    if net >= delta:
        return 1
    else:
        return -1


def treinamento(X, y, learning_rate, max_it, threshold_error, activation):
    print(
        f'Treinando o modelo com a função de ativação {highlight(activation.__name__)}')

    # Definir uma seed para o gerador de números aleatórios do numpy
    seed = np.random.RandomState(156482156)

    # Inicializar o vetor w de pesos e o bias
    w = seed.rand(X.shape[1])
    print(f'Pesos Iniciais: {highlight(w)}\n')

    # Inicia o  Valor aleatório
    bias = seed.rand()

    t = 1
    soma_erro = 0

    while t < max_it and soma_erro <= threshold_error:
        soma_erro = 0
        for index, row in X.iterrows():
            # Calcular o valor de net
            # Utilizamos a função sum() do python
            # que soma os valores retornados por um iterador
            # Passamos como iterador um laço for,
            # que pra cada valor i nos pesos w e pra cada linha j, multiplica i * j
            # Por fim somamos o bias
            net = sum(wi * xi for wi, xi in zip(w, row)) + bias

            # Descobrimos o valor predizido utilizando a função de ativação
            y0 = activation(net, delta=DELTA)

            # Calculamos o erro e o erro²
            erro = y[index] - y0
            soma_erro = soma_erro + erro ** 2

            # Por fim atualizamos os valores dos pesos w
            w_att = []
            i = 0
            for wi in w:
                # Adequação de cada peso
                new_w = wi - learning_rate * erro * (-row[i])
                w_att.append(new_w)
                i += 1
            w = np.array(w_att)

            # E atualizamos o valor do bias
            bias = bias - learning_rate * erro * (-1)
        print(f'Época {highlight(t)}')
        print(f'Pesos: {highlight(w)}\n')
        t = t + 1

    return w, bias


if __name__ == '__main__':
    header('Treinando modelos de Perceptron')

    title('Conjunto de dados OR')
    df, labels = load_dataframe('./data/or.csv')

    learning_rate = 0.01
    max_iterations = 10
    threshold_error = 1000

    w, bias = treinamento(df, labels, learning_rate,
                          max_iterations, threshold_error, binary_degree)

    print(highlight("Veto de pesos"))
    print(w)
    print(highlight("Bias"))
    print(bias)

    title('Conjunto de dados Claro-Escuro')
    df, labels = load_dataframe('./data/claro-escuro.csv')

    learning_rate = 0.01
    max_iterations = 10
    threshold_error = 1000

    w, bias = treinamento(df, labels, learning_rate,
                          max_iterations, threshold_error, bipolar_degree)

    print(highlight("Veto de pesos"))
    print(w)
    print(highlight("Bias"))
    print(bias)
