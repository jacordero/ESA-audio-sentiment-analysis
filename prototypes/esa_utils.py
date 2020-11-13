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
import resource
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
    raise Exception("NO DIRECTORY FOUND")

def get_time():
    """Get the current local time in the Netherlands.

    Returns:
      The current local time.

    """
    tz_nl = pytz.timezone('Europe/Amsterdam')
    time_nl = datetime.now(tz_nl)
    return time_nl.strftime("%H_%M_%S")

def logging(data, logging_file_name):
    """Write the data to logging file.

    Args:
      data: list from prediction of models.
      logging_file_name: A name of logging file.

    """  
    folder_name  = os.path.join(get_path('docs'), logging_file_name + '_' + str(date.today()))
    
    os.makedirs(folder_name, exist_ok=True)
      
    # create the file if not exist and write data to file.
    with open(os.path.join(folder_name, logging_file_name + '_logging_' + get_time() + '.json'), 'w+') as logging_file:
      data.insert(0, [datetime.now().strftime('%d-%m-%y %H:%M:%S')])
      json.dump(data, logging_file)
  
def get_temperature():
    """Measure the operating temperature of pi.

       The maximum recommended operating temperature is 80 'C. 
       If the operating temperature cross 85 'C, CPU starts throttling and reduces 
       the clock to cool down the temperature. This will impact the performance.

    Returns:
      The current temperature of .

    """
    temp = os.popen("vcgencmd measure_temp").readline()
    temp = temp.replace("temp=", "")
    if temp is "85'C":
      print("WARNING: pi operating temperature is {}".format(temp) )

    return (temp.replace("temp=", ""))

def get_pi_resource_info(process):
    """Retrieve resource usage information by a process.

    Args:
      process: A name of process.

    """
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)

    before = usage
    print(subprocess.run(['python', process],  capture_output=True))
    after = usage

    print("==================================================================")
    print("User CPU time = {} seconds.".format(after[0] - before[0]))
    print("System CPU time = {} seconds.".format(after[1] - before[1]))
    print("==================================================================")
    print("MEMORY = {} kilobytes.".format(after[2] - before[2]))
    print("==================================================================")
    print("TEMPERATURE: {}".format(get_temperature()))

