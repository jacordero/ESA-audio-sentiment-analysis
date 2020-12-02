import pytest
import yaml
import os
import json
import sys
from src.stern_utils import Utils
import time
from datetime import datetime, date
from datetime import timedelta


def create_instance(data, logging_file_name):
    Utils.logging(data, logging_file_name)

def create_file_path(logging_file_name):
    root_folder = os.path.dirname(os.path.normpath(os.getcwd()))
    folder_name = os.path.join(root_folder, logging_file_name + '_' + str(date.today()))
    log_file_name = os.path.join(folder_name, 'log_' + Utils.get_time() + '.json')
    return log_file_name


def read_file(file_name):
    with open(file_name) as input_file:
        log_file = json.load(input_file)
    print(len(log_file))
    print(log_file[len(log_file)-1]["tone_emotion"])
    return log_file

def diff_content(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif

@pytest.mark.parametrize("data,logging_file_name", [
    ([['happy:0.00', 'sad:0.09', 'angry:0.30']], "audio-sentiment-analysis\\logs\\test_logging_file")
])
def test_timestamp_creation(data, logging_file_name):
    create_instance(data, logging_file_name)
    log_file_name = create_file_path(logging_file_name)
    log_file  = read_file(log_file_name)
    assert log_file[len(log_file)-1]["time"]
   

@pytest.mark.parametrize("data,logging_file_name", [
    ([['happy:0.00', 'sad:0.09', 'angry:0.30']], "audio-sentiment-analysis\\logs\\test_logging_file")])
def test_timestamp(data, logging_file_name):
    current_time = datetime.now()
    time_str = current_time.strftime('%d-%m-%y %H:%M')
    create_instance(data, logging_file_name)
    log_file_name = create_file_path(logging_file_name)
    log_file  = read_file(log_file_name)
    timestamp = log_file[len(log_file)-1]["time"]
    time_details = timestamp.split(":")
    date_hour = time_details[0]
    minutes = time_details[1]
    timestamp_without_sec  = date_hour + ":" + minutes
    if minutes != "59":
        new_time = current_time + timedelta(minutes = 1)        
    else:
        new_time = current_time + timedelta(hours = 1)
    assert timestamp_without_sec == time_str or timestamp_without_sec == new_time.strftime('%d-%m-%y %H:%M')
