import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from style import header, title
from utils import load_dataframe


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


def main():
    header('Leitura e análise do conjunto de dados SPAM')

    df, labels = load_dataframe('./data/spambase.csv')

    best_features = [
        "word_freq_you",
        "charfreq!",
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total"
    ]

    title('Alguns dados')
    print(df[best_features])

    title('Quantidade de missing values por coluna')
    print(df.isnull().sum())

    title('Análises sobre o algumas colunas')
    print(df[best_features].describe())

    balanced_df, balanced_labels = balance(df, labels)

    title('Quantidade de linhas e colunas antes do balanceamento')
    print(df.shape)

    title('Quantidade de linhas e colunas depois do balanceamento')
    print(balanced_df.shape)

    normalized_df = normalize(balanced_df)

    title('Alguns dados após a normalização por reescala linear')
    print(normalized_df[best_features])

    title('Análises sobre os dados finais')
    print(normalized_df[best_features].describe())


if __name__ == '__main__':
    main()
