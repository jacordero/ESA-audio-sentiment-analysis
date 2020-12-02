"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved. 
@ Authors: Raha Sadeghi r.sadeghi@tue.nl; George Azis g.azis@tue.nl;
Last modified: 02-12-2020
"""

import pytest
import os
import json
from src.stern_utils import Utils
from datetime import datetime, date

def create_instance(data, logging_file_name):
    """This function calls the logging method of stern_utils

    Args:
        data: the directory contatining the model
        logging_file_name: the path to the logging file and the prefix for log file name
    """
    Utils.logging(data, logging_file_name)

def create_file_path(logging_file_name):
    """This function creates the path and log file name

    Args:
        logging_file_name: the path to the logging file and the prefix for log file name

    Returns:
        the complete path to the log file
    """
    root_folder = os.path.dirname(os.path.normpath(os.getcwd()))
    folder_name = os.path.join(root_folder, logging_file_name + '_' + str(date.today()))
    log_file_name = os.path.join(folder_name, 'log_' + Utils.get_time() + '.json')
    return log_file_name

@pytest.mark.parametrize("data,logging_file_name", [
    ([['happy:0.00', 'sad:0.09', 'angry:0.30']], "audio-sentiment-analysis\\logs\\test_logging_file")
])
def test_file_creation(data, logging_file_name):
    """This function tests whether a new file can be created if not exists already

    Args:
        data: the directory contatining the model
        logging_file_name: the path to the logging file and the prefix for log file name
    """
    log_file_name = create_file_path(logging_file_name)
    if not os.path.exists(log_file_name):
        create_instance(data, logging_file_name)
        assert os.path.exists(log_file_name)

