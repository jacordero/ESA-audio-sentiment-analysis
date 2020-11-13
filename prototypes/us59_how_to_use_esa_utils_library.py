"""
Copyright (C) Tu/e.

@author Tuvi Purevsuren t.purevsuren@tue.nl

======================================================

This script is used to demonstrate the esa_utils library.

"""
import os
import esa_utils as eu

def main(ARGS):
    """Execute the demonstration of esa utils library.

    Args:
      ARGS: A dictionary of parameters.

    """
    data = [ ['Angery'], ['I hate you.'], ['emotion distribution']]

    eu.logging(data, ARGS['logging_file_name'])
    eu.get_pi_resource_info(ARGS['process_name'])

if __name__ == '__main__':
    data = [ ['Angery'], ['I hate you.'], ['emotion distribution']]
    ARGS = {
        'logging_file_name': 'testing_utils',
        'process_name': 'us19_tone_based_emotion_detection_rasberry_pi.py'
    }
    main(ARGS)