import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler


def one_on_one_plot(var_1: str, var_2: str, df: pd.DataFrame):
    rate = 1.01
    data_pairplot = pd.concat([df[var_2], df[var_1]], axis=1)
    plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var_1, y=var_2, data=data_pairplot)
    fig.axis(ymin=data_pairplot[var_2].min() / rate, ymax=rate * data_pairplot[var_2].max())


def correlation(df: pd.DataFrame, main_col):
    corr = df.corr(numeric_only=True)[main_col]
    sorted_correlation = corr.sort_values(ascending=False)
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):  # more options can be specified also
        print("Correlation:\n{}".format(sorted_correlation))


def corrupt(df, col_to_corrupt, frac):
    for i in df[col_to_corrupt].sample(frac=frac, random_state=1).index:
        df.loc[i, col_to_corrupt] = np.random.randint(1, 11, 1)[0]


def pair_plot(df, col_to_plot_against):
    data_num = df.select_dtypes(include=['float64', 'int64']).columns
    for col in data_num:
        df[col] = df[col].fillna(0)

    for i in range(0, len(data_num), 5):
        sns.pairplot(data=df,
                     x_vars=data_num[i:i + 5],
                     y_vars=[col_to_plot_against])


def one_hot_encode(df, df_cat_nominal):
    """
    One-hot encode nominal categorical variable.
    """
    df = df.drop(df_cat_nominal.columns, axis=1)
    return df.join(pd.get_dummies(df_cat_nominal, drop_first=True))


def count_distinct(df_cat_nominal, df):
    """

    :param df:
    :param df_cat_nominal:
    :return:
    """
    data_cat_col_dict = {}
    for col in df_cat_nominal:
        data_cat_col_dict[col] = (Counter(df[col]))
    return data_cat_col_dict


def standardization(df):
    """
    :param df: dataframe to normalize
    :return: normalized dataframe
    """
    df_num = df.select_dtypes(include=['float64', 'int64']).columns
    df[df_num] = df[df_num].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    return df


def normalization(df):
    scaler = MinMaxScaler()
    df_num = df.select_dtypes(include=['float64', 'int64']).columns
    df[df_num] = scaler.fit_transform(df[df_num])
    return df
