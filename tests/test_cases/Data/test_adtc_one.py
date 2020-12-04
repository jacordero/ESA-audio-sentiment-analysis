"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved. 
@ Authors: Raha Sadeghi r.sadeghi@tue.nl; George Azis g.azis@tue.nl;
Last modified: 02-12-2020
"""
import pytest
import os
from pathlib import Path



def test_sequential_data_existence():
    data_directory="prod_data\\sequential\\test"
    test_mfcc_data = os.path.join(data_directory, "mfcc_test.joblib")
    test_labels_data = os.path.join(data_directory, "labels_test.joblib")
    data_exist = True
    if not os.path.exists(test_mfcc_data) or not os.path.exists(test_labels_data):
        data_exist = False
    assert data_exist




def test_siamese_data_existence():
    data_directory="prod_data/siamese/test"
    test_mfcc_data = os.path.join(data_directory, "mfcc_test.joblib")
    test_labels_data = os.path.join(data_directory, "labels_test.joblib")
    test_lmfe_data = os.path.join(data_directory, "lmfe_test.joblib")
    data_exist = True
    if not os.path.exists(test_mfcc_data) or not os.path.exists(test_labels_data) or not os.path.exists(test_lmfe_data):
        data_exist = False
    assert data_exist


