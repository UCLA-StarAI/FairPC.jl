import numpy as np
import pandas as pd
import argparse
import sys
import os
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity
from data import *


def reduction(name, fold, num_X=0):
    # load data
    train_data, test_data, cloumns, _, train_y_fair, train_y_proxy, test_y_fair, test_y_proxy = load_data(name, fold, num_X=num_X, use_fair=False)

    train_X = train_data.drop(columns=cloumns)
    train_y_true = train_data[DATA2D[name]]
    train_sex = train_data[DATA2S[name]]

    # learn
    learn = ExponentiatedGradient(
        LogisticRegression(solver='liblinear', fit_intercept=True),
        constraints=DemographicParity())

    learn.fit(train_X, train_y_true, sensitive_features=train_sex)

    # predict
    test_X = test_data.drop(columns=cloumns)
    prob_test = learn._pmf_predict(test_X)[:, 1]
    prob_train = learn._pmf_predict(train_X)[:, 1]
    s_train = train_sex.astype(bool)
    s_test = np.array(test_data[DATA2S[name]]).astype(bool)


    save_file(name, num_X, fold, "Reduction", prob_train, s_train, train_y_fair, train_y_proxy, prob_test, s_test, test_y_fair, test_y_proxy)


def main():
    name, fold, num_X, _ = read_cmd()
    reduction(name, fold, num_X=num_X)


if __name__ == "__main__":
    main()
