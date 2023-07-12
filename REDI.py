from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
import shap
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import re


def get_score(model, feature, target):
    pred = model.predict(feature)
    score = pd.DataFrame(pred, columns=['Prediction'], index=feature.index)
    score['GroundTruth'] = np.array(target)
    score['AbsoluteError'] = abs(score['GroundTruth'] - score['Prediction'])
    return score


def get_doubtful_values(score, error=1.5):
    return score[score['AbsoluteError'] > error]


def get_truthful_values(score, error=0.5):
    return score[score['AbsoluteError'] < error]


class REDI:
    def __init__(self, data: pd.DataFrame):
        self.y_score = None
        self.y_pred = None
        self.ridge = RidgeCV()
        self.X = None
        self.y = None
        self.data = data

    def train_model(self, X, y):
        self.y = self.data["OverallQual"]
        self.X = self.data.drop(["OverallQual", 'SalePrice'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=100, stratify=y)
        self.ridge.fit(X_train, y_train)

    def get_prediction(self):
        self.y_pred = self.ridge.predict(self.X)
        self.y_score = get_score(self.ridge, self.X, self.y)

    def loc_doubtful_values(self):
        return get_doubtful_values(self.y_score, error=1)

    def loc_truthful_values(self):
        return get_truthful_values(self.y_score, error=0.5)

    def get_truthful_data(self, y_correct):
        data_correct = self.data.iloc[y_correct.index]
        data_correct['data_index'] = data_correct.index
        data_correct = data_correct.reset_index(drop=True)
        data_correct['index'] = data_correct.index
        data_correct.set_index(['index', 'data_index'], inplace=True)

        return data_correct

    def corrupt_data(self, data_to_corrupt):
        corruption_list = []
        old_new_value = OrderedDict()
        OverallQual_sample_fraction = 0.05
        corruption_index_list = sorted(data_to_corrupt.sample(frac=OverallQual_sample_fraction, random_state=1).index)
        for i in corruption_index_list:
            old = data_to_corrupt.loc[i, 'OverallQual']
            new = old - 6 if old > 6 else old + 6
            if new > 10: new = 10
            # while abs(old - new) != 6:
            #     new = np.random.randint(1, 11, 1)[0]
            data_to_corrupt.loc[i, 'OverallQual'] = new
            corruption_list.append(i)
            old_new_value[i[0]] = (old, new)
        print(len(corruption_list))




        # y_corrupted = data_to_corrupt["OverallQual"]
        # X_corrupted = data_to_corrupt.drop(["OverallQual", 'SalePrice'], axis=1)
        #
        # X_train_cor, X_test_cor, y_train_cor, y_test_cor = train_test_split(X_corrupted, y_corrupted, test_size=0.1,
        #                                                                     random_state=100)
        # ridge_corrupt = RidgeCV()
        # ridge_corrupt.fit(X_train_cor, y_train_cor)
        # print(ridge_corrupt.score(X_test_cor, y_test_cor))
        # # %%

