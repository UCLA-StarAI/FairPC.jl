import os
import sys
import argparse
import numpy as np
import pandas as pd
import math
from random import random
from random import seed
from data import *

def main(name, fold, num_X=None, use_fair=False):
    train_data, test_data, cloumns, decision_label, train_y_fair, train_y_proxy, test_y_fair, test_y_proxy = load_data(name, fold, num_X=num_X, use_fair=use_fair)
    
    s_train = np.array(train_data[DATA2S[name]])
    s_test = np.array(test_data[DATA2S[name]])

    prob_train = [random() for x in s_train]

    prob_test = [random() for x in s_test]

    save_file(name, num_X, fold, "Random", prob_train, s_train, train_y_fair, train_y_proxy, prob_test, s_test, test_y_fair, test_y_proxy)


if __name__ == "__main__":
    name, fold, num_X, use_fair = read_cmd()
    main(name, fold, num_X, use_fair)
