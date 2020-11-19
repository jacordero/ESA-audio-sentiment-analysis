"""
Copyright (C) Tu/e.

@author Tuvi Purevsuren t.purevsuren@tue.nl

======================================================

This is the util library.

"""
import os
import json 
import time
import pytz
import subprocess

from datetime import datetime, date


def get_path(dir_name):
    """Find path of requested directory such as 'models' and 'data' and 'docs'.

    Args:
       dir_name: A name of directory.

    Returns:
      The full path of requested directory.

    """
    path = os.path.join(os.path.dirname(os.path.normpath(os.getcwd())), dir_name)
    if path: 
      return path
    else:
      raise Exception("NO DIRECTORY FOUND")

def get_time():
    """Get the current local time in the Netherlands.

    Returns:
      The current local time.

    """
    tz_nl = pytz.timezone('Europe/Amsterdam')
    time_nl = datetime.now(tz_nl)
    return time_nl.strftime("%H")

def logging(data, logging_file_name):
    """Write the data to logging file.

    Args:
      data: list parameters for example [["13-11-20 09:53:12"], ["emotion"], ["emotion distribution"], ["text"]].
      logging_file_name: A name of logging file.

    """  
    folder_path = os.path.join(os.getcwd(), 'src/logs')
    folder_name  = os.path.join(folder_path, logging_file_name + '_' + str(date.today()))
    # if directory does not exist , create
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)
    if not os.path.exists(folder_name):
      os.makedirs(folder_name)

    log_input = {
      'time': datetime.now().strftime('%d-%m-%y %H:%M:%S'), 
      'text': data[0], 
      'tone_emotion': data[1], 
      'text_emotion': data[2]
       }

    log_file_name = os.path.join(folder_name, 'log_' + get_time() + '.json')
    if os.path.exists(log_file_name):
      with open(log_file_name, 'a') as logging_file:
        logging_file.write(log_input)
        #temp = json.load(logging_file)
        #temp.append(log_input)
        #with open(log_file_name, 'w') as logging_file:
        #  json.dump(temp, logging_file, indent=2)
    else:
      # create the file if not exist and write data to file.
      with open(log_file_name, 'w+') as logging_file:
        json.dump(log_input, logging_file, indent=2)
        
