import os
import sys
import argparse
import numpy as np
import pandas as pd
sys.path.insert(0, './fair_classification/') # the code for fair classification is in this directory
import utils as ut
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints
import math
from data import *


def model(x_train, y_train, x_control_train, x_test, y_test, x_control_test, SV):
    x_train = ut.add_intercept(x_train)
    x_test = ut.add_intercept(x_test)
    apply_fairness_constraints = 1
    apply_accuracy_constraint = 0
    sensitive_attrs = [SV]
    sensitive_attrs_to_cov_thresh = {SV: 0}
    sep_constraint = 0
    loss_function = lf._logistic_loss
    gamma = 0
    
    def train_test_classifier():
        w = ut.train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
        train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w, x_train, y_train, x_test, y_test, None, None)
        distances_boundary_test = np.dot(x_test, w)
        distances_boundary_train = np.dot(x_train, w)
        prob_test = [sigmoid(x) for x in distances_boundary_test]
        prob_train = [sigmoid(x) for x in distances_boundary_train]
        all_class_labels_assigned_test = np.sign(distances_boundary_test)
        correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
        cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs)
        p_rule = ut.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])	
        # return w, p_rule, test_score
        return prob_train, prob_test
    return train_test_classifier()


def main(name, fold, num_X=None, use_fair=False):

    train_data, test_data, cloumns, decision_label, train_y_fair, train_y_proxy, test_y_fair, test_y_proxy = load_data(name, fold, num_X=num_X, use_fair=use_fair)

    cloumns.append(DATA2S[name])
    x_train = np.array(train_data.drop(columns=cloumns))
    y_train = np.array(train_data[decision_label])
    y_train = np.array([-1 if y == 0 else 1 for y in y_train])
    S_train = np.array(train_data[DATA2S[name]])
    SV = DATA2S[name]
    x_control_train = {SV: S_train}


    x_test = np.array(test_data.drop(columns=cloumns))
    y_test = np.array(test_data[decision_label])
    y_test = np.array([-1 if y == 0 else 1 for y in y_test])
    S_test = np.array(test_data[DATA2S[name]])
    x_control_test = {SV: S_test}

    prob_train, prob_test = model(x_train, y_train, x_control_train, x_test, y_test, x_control_test, SV)

    save_file(name, num_X, fold, "LR", np.array(prob_train), S_train, train_y_fair, train_y_proxy, np.array(prob_test), S_test, test_y_fair, test_y_proxy)


if __name__ == "__main__":
    name, fold, num_X, use_fair = read_cmd()
    main(name, fold, num_X, use_fair)
