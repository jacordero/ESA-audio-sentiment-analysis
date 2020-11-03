import os
import fnmatch
import csv

import pandas as pd
import pyreadstat

def read_SPSS_file(sav_file_path):
    print("SPSS FILE")
    file_path = os.path.join(sav_file_path, 'isear_spss.sav')
    dt, meta = pyreadstat.read_sav(file_path)
    print(type(dt))
    print(dt.head())
    print(dt.tail())


def create_datasets_for_esa_model():
    print("================ CSV FILE ====================")

    # input and output directories for datasets
    input_dir = os.path.join(os.getcwd(), 'input_datasets')
    output_dir = os.path.join(os.getcwd(), 'output_datasets')

    if not os.path.isdir(input_dir):
        print("DIRECTORY NOT FOUND")
    else:
        # to read .sav file
        #read_SPSS_file(input_dir)
        
        input_files = os.listdir(input_dir)
        print(input_files)
        
        # open the file esa_datasets with 'w' write permission.
        with open(os.path.join(output_dir, 'esa_datasets.csv'), 'w', newline='') as esa_datasets:
            # [index , text, emotion] - sequence of the keys to write to file
            index = 0
            writer = csv.writer(esa_datasets)
            writer.writerow(['index', 'text', 'emotion'])
            for file_name in input_files:
                print(file_name)
            
                # file name string matches the pattern string e.g : 'train_sent_emo.csv'
                if fnmatch.fnmatch(file_name, 'train_sent_emo.csv'):
                    with open(os.path.join(input_dir, file_name), mode='r') as input_data:
                        reader = csv.reader(input_data)
                        # to read the header
                        header = next(reader)
                        for rows in reader:
                            writer.writerow([index, rows[1], rows[3]])
                            index +=1
                # file name string matches the pattern string e.g: 'ISEAR_Dataset.csv'
                if fnmatch.fnmatch(file_name, 'ISEAR_Dataset.csv'):
                    with open(os.path.join(input_dir, file_name), mode='r',  encoding='latin1') as input_data:
                        reader = csv.reader(input_data)
                        # to read the header
                        header = next(reader)
                        for rows in reader:
                            if (rows[0] != 'shame') and (rows[0] != 'guilt'):
                                writer.writerow([index, rows[1], rows[0]])
                                index +=1
                                

def main():
    print("==== WELCOME to the TEXT SENTIMENT ====")
    create_datasets_for_esa_model()

if __name__ == "__main__":
    main()
