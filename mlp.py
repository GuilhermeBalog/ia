import numpy as np
from math import exp


def f(net):
    result = []
    for i in net:
        result.append((1 / (1 + exp(-i))))

    return result


def df_dnet(f_net):
    return (f_net * (1 - f_net))


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


if __name__ == '__main__':
    model = architecture()

    result = forward(model, [1, 2])
    print(result)
