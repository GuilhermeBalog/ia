import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import sys

#  text       background

# 30 black    preto    40
# 31 red      vermelho 41
# 32 green    verde    42
# 33 yellow   amarelo  43
# 34 blue     azul     44
# 35 Magenta  Magenta  45
# 36 cyan     ciano    46
# 37 grey     cinza    47
# 97 white    branco   107


def load_dataframe():
    df = pd.read_csv('./data/spambase.csv')

    labels = df['label'].to_numpy()
    # remove coluna das labels
    df = df.drop('label', axis=1)

    return df, labels


def normalize(df):
    normalized_df = df.copy()

    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        distance = max_value - min_value
        normalized_df[feature_name] = (df[feature_name] - min_value) / distance

    return normalized_df


def balance(df, labels):
    return SMOTE(random_state=12345).fit_resample(
        df, labels)


def title(text):
    text = ' {} '.format(text)
    if sys.platform.startswith('linux'):
        print('\033[1;32m\n\n{}'.format(text.upper()))
        print('-'*len(text), end='\n\n\033[m')
    else:
        print('\n\n{}'.format(text.upper()))
        print('-'*len(text), end='\n\n')

def header(text):
    if sys.platform.startswith('linux'):
        print('\033[1;31m-\033[0;34m=\033[1;31m-\033[0m' * 20)
        print('\033[1;31m{:^60}\033[m'.format(text))
        print('\033[1;31m-\033[0;34m=\033[1;31m-\033[0m' * 20)
    else:
        print('-=-' * 20)
        print('{:^60}'.format(text))
        print('-=-' * 20)


def main():
    header('Leitura e análise do conjunto de dados SPAM')

    df, labels = load_dataframe()

    best_labels = [
        "word_freq_you",
        "charfreq!",
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total"
    ]

    title('Colunas')
    print(df.info(memory_usage=False, verbose=True))

    title('Alguns dados')
    print(df[best_labels].head())

    title('Análises sobre o algumas colunas')
    print(df[best_labels].describe())

    balanced_df, balanced_labels = balance(df, labels)

    title('Quantidade de linhas e colunas depois do balanceamento')
    print(balanced_df.shape)

    normalized_df = normalize(balanced_df)

    title('Alguns dados após a normalização por reescala linear')
    print(normalized_df[best_labels].head())

    title('Análises sobre os dados finais')
    print(normalized_df[best_labels].describe())


if __name__ == '__main__':
    main()
