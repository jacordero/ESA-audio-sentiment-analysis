import pytest
import yaml
import os
import json
from src.stern_utils import Utils
import time
from datetime import datetime, date

def create_instance(data, logging_file_name):
    Utils.logging(data, logging_file_name)

def create_file_path(logging_file_name):
    root_folder = os.path.dirname(os.path.normpath(os.getcwd()))
    folder_name = os.path.join(root_folder, logging_file_name + '_' + str(date.today()))
    log_file_name = os.path.join(folder_name, 'log_' + Utils.get_time() + '.json')
    return log_file_name

@pytest.mark.parametrize("data,logging_file_name", [
    ([], "audio-sentiment-analysis\\logs\\test_logging_file"),
    (None, "audio-sentiment-analysis\\logs\\test_logging_file"),
    ([['happy:0.00', 'sad:0.09', 'angry:0.30']], "audio-sentiment-analysis\\logs\\test_logging_file"), 
     
])
def test_file_creation(data, logging_file_name):
    create_instance(data, logging_file_name)
    log_file_name = create_file_path(logging_file_name)
    print(log_file_name)
    if data is None or len(data) == 0:
        assert not os.path.exists(log_file_name)
    else:
        assert os.path.exists(log_file_name)



