"""
Copyright (C) Tu/e.

@author Tuvi Purevsuren t.purevsuren@tue.nl

This script monitors the resource utilized by a process. 

Current system status CPU, RAM, and temperature.

======================================================

"""

import sys
import os
import resource
import subprocess

def get_temperature():
    """Measure the operating temperature of pi.

       The maximum recommended operating temperature is 80 'C. 
       If the operating temperature cross 85 'C, CPU starts throttling and reduces 
       the clock to cool down the temperature. This will impact the performance.

    Returns:
      The current temperature of .

    """
    temp = os.popen("vcgencmd measure_temp").readline()
    return (temp.replace("temp=", ""))

def get_memory_info():
    """Retrieve resource usage information by a process.

    Returns:
      An object that describes the resources consumed by a process.

    """
    usage = resource.getrusage(resource.resource.RUSAGE_CHILDREN)
    return usage

def monitor_resource(process):
    """Monitor the process usage.

    Args:
      process: A name of process.

    """
    before = get_memory_info()
    print(subprocess.run(['python', process], capture_output=True))    
    after = get_memory_info()
    
    print("==================================================================")
    print("User CPU time = {} seconds.".format(after[0] - before[0]))
    print("System CPU time = {} seconds.".format(after[1] - before[1]))
    print("==================================================================")
    print("MEMORY = {} kilobytes.".format(after[2] - before[2]))
    print("==================================================================")
    print("TEMPERATURE: {}".format(get_temperature()))

if __name__ == "__main__":
    print("==== WELCOME Raspberry pi: RESOURCE PROFILE ====")
    process_name = sys.argv[1]
    
    monitor_resource(process_name)