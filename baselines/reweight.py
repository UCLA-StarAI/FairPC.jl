import numpy as np
import pandas as pd
import argparse
from sklearn.linear_model import LogisticRegression
from data import *

def debias_weights(original_labels, protected_attributes, multipliers):
  exponents = np.zeros(len(original_labels))
  for i, m in enumerate(multipliers):
    exponents -= m * protected_attributes[i]
  weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
  weights = np.where(original_labels > 0, 1 - weights, weights)
  return weights

def get_error_and_violations(y_pred, y, protected_attributes):
  acc = np.mean(y_pred != y)
  violations = []
  for p in protected_attributes:
    protected_idxs = np.where(p > 0)
    violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
  pairwise_violations = []
  for i in range(len(protected_attributes)):
    for j in range(i+1, len(protected_attributes)):
      protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
      if len(protected_idxs[0]) == 0:
        continue
      pairwise_violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
  return acc, violations, pairwise_violations


def learning(X_train, y_train, X_test, y_test, protected_train, protected_test):
    multipliers = np.zeros(len(protected_train))
    learning_rate = 1.
    n_iters = 100
    for it in range(n_iters):
        weights = debias_weights(y_train, protected_train, multipliers)
        model = LogisticRegression()

        model.fit(X_train, y_train, weights)
        y_pred_train = model.predict(X_train)
        acc, violations, pairwise_violations = get_error_and_violations(y_pred_train, y_train, protected_train)
        multipliers += learning_rate * np.array(violations)


    if (it + 1) % n_iters == 0:
        print(multipliers)
        y_pred_test = model.predict(X_test)
        test_scores = model.decision_function(X_test)
        train_scores = model.decision_function(X_train)
        prob_test = np.array([sigmoid(x) for x in test_scores])
        prob_train = np.array([sigmoid(x) for x in train_scores])

        acc, violations, pairwise_violations = get_error_and_violations(y_pred_train, y_train, protected_train)
        print("Train Accuracy", acc)
        print("Train Violation", max(np.abs(violations)), " \t\t All violations", violations)
        if len(pairwise_violations) > 0:
            print("Train Intersect Violations", max(np.abs(pairwise_violations)), " \t All violations", pairwise_violations)

        acc, violations, pairwise_violations = get_error_and_violations(y_pred_test, y_test, protected_test)
        print("Test Accuracy", acc)
        print("Test Violation", max(np.abs(violations)), " \t\t All violations", violations)
        if len(pairwise_violations) > 0:
            print("Test Intersect Violations", max(np.abs(pairwise_violations)), " \t All violations", pairwise_violations)
        print()
        print()
        return prob_train, prob_test


def model(name, fold, num_X=None):
    train_data, test_data, cloumns, learn_decision_label, train_y_fair, train_y_proxy, test_y_fair, test_y_proxy = load_data(name, fold, num_X=num_X, use_fair=False)

    # load_data
    X_train = np.array(train_data.drop(columns=cloumns))
    y_train = np.array(train_data[learn_decision_label])
    X_test = np.array(test_data.drop(columns=cloumns))
    y_test = np.array(test_data[learn_decision_label])
    
    s_train = np.array(train_data[DATA2S[name]])
    protected_train = [s_train]
    s_test = np.array(test_data[DATA2S[name]])
    protected_test = [s_test]



    prob_train, prob_test = learning(X_train, y_train, X_test, y_test, protected_train, protected_test)
    #print(1 - accuracy(prob, true_y_proxy))
    save_file(name, num_X, fold, "Reweight", prob_train, s_train, train_y_fair, train_y_proxy, prob_test, s_test, test_y_fair, test_y_proxy)


def main():
    name, fold, num_X, _ = read_cmd()
    model(name, fold, num_X=num_X)


if __name__ == "__main__":
    main()