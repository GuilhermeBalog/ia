import numpy as np
import pandas as pd
from math import exp


def f(net):
    result = []
    for i in net:
        result.append((1 / (1 + exp(-i))))

    return result


def df_dnet(f_net):
    result = []
    for i in f_net:
        result.append(i * (1 - i))
    return result


def random_weights_matrix(neurons, inputs):
    columns = (inputs + 1)
    rows = neurons
    weights_length = columns * rows
    random_weight_list = np.random.default_rng().standard_normal(
        weights_length).reshape(rows, columns)

    return random_weight_list


def architecture(input_length=2, hidden_length=2, output_length=1, activation_function=f, d_activation_function=df_dnet):
    hidden = random_weights_matrix(hidden_length, input_length)
    output = random_weights_matrix(output_length, hidden_length)

    return {
        "input_length": input_length,
        "hidden": hidden,
        "hidden_length": hidden_length,
        "output": output,
        "output_length": output_length,
        "f": activation_function,
        "df": d_activation_function
    }


def forward(model, sample):
    sample.append(1)
    net_hidden = np.zeros((1, len(model["hidden"])))
    np.matmul(model["hidden"], sample, net_hidden)

    # Transforma pra vetor
    net_hidden = net_hidden[0]

    f_net_hidden = model["f"](net_hidden)

    f_net_hidden.append(1)
    net_output = np.zeros((1, len(model["output"])))
    np.matmul(model["output"], f_net_hidden, net_output)
    net_output = net_output[0]
    f_net_output = model["f"](net_output)

    return {
        "net_hidden": net_hidden,
        "net_output": net_output,
        "f_net_hidden": f_net_hidden,
        "f_net_output": f_net_output
    }


def backpropagation(model, dataset, eta=0.1, treshold=0.001):
    derivative_function = model["df"]
    squared_error = 2 * treshold

    counter = 0
    while squared_error > treshold:
        squared_error = 0

        for i, row in dataset.iterrows():
            xi = row[:model["input_length"]]
            yi = row[model["input_length"]:]

            results = forward(model, xi)

            oi = results["f_net_output"]

            error = []
            total_squared_error = 0
            for j in range(len(oi)):
                error[j] = yi[j] - oi[j]
                total_squared_error += error[j] ** 2

            squared_error += total_squared_error

            derivative_result = derivative_function(results["f_net_output"])
            delta_oi = []
            for j in range(len(error)):
                delta_oi[j] = error[j] * derivative_result[j]

    return


if __name__ == '__main__':
    model = architecture()
    df = pd.read_csv('./data/iris.csv')

    #result = forward(model, [1, 2])
    # print(result)
