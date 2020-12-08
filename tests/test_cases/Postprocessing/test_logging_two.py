"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved. 
@ Authors: Raha Sadeghi r.sadeghi@tue.nl; George Azis g.azis@tue.nl;
Last modified: 02-12-2020
"""
import pytest
import os
import json
import sys
from src.stern_utils import Utils
import time
from datetime import datetime, date
from datetime import timedelta

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

def read_file(file_name):
    """This function creates the path and log file name

    Args:
        logging_file_name: the path to the logging file and the prefix for log file name

    Returns:
        the complete path to the log file
    """
    with open(file_name) as input_file:
        log_file = json.load(input_file)
    return log_file

def diff_content(list1, list2):
    """This function creates the path and log file name

    Args:
        list1: the first list to be compared with the second list
        list2: the second list to be compared with the first list

    Returns:
        a list containing the difference of two lists
    """
    li_dif = [i for i in list1 + list2 if i not in list1 or i not in list2]
    return li_dif

@pytest.mark.parametrize("data,logging_file_name", [
    ([['happy:0.00', 'sad:0.09', 'angry:0.30']], "audio-sentiment-analysis\\logs\\test_logging_file")
])
def test_file_content(data, logging_file_name):
    """This function tests if data has been written correctly in the log file

    Args:
        data: the directory contatining the model
        logging_file_name: the path to the logging file and the prefix for log file name
    """
    create_instance(data, logging_file_name)
    log_file_name = create_file_path(logging_file_name)
    log_file  = read_file(log_file_name)
    diff = diff_content(log_file[len(log_file)-1]["tone_emotion"] , data[0] )
    assert not diff


