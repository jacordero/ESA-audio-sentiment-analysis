# Running us15_model_to_run_in_raspberry.py on the Raspberry Pi

To run the script, some preparations needed to be made:

## Loading the data and models

The data and models used in the script (and any other script for that matter) are not and will not be stored in this repository. They are instead stored in the SharePoint. Specifically, the data and models can be found [here](https://tuenl.sharepoint.com/:f:/r/sites/gad_cbo/JPC/MC/ESA%20PDEng%20ST%20Project/ModelsAndData/Audio/Development?csf=1&web=1&e=Tflpy2). The repository does contain (empty) folders with the same name as the folders in the SharePoint. All code expects data and models to be located in this folder when you execute it. To use the data and models stored in the SharePoint, you have to copy the contents and of the `data` folder in the SharePoint into the `data` folder of your locally cloned copy of the git repository. The same should be done with the `models` folders in both locations.  

## Installation of dependencies

The script has quite a few dependencies. These dependencies are already installed in a Python Virtual Environment to avoid cluttering the Raspberry Pi's system libraries (https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/). This environment is located in the folder `~/Documents/esa/experiments/audio-sentiment-analysis`. 

The following commands can be used to install all dependencies of the script:

```bash
# (https://stackoverflow.com/a/48244595)
sudo apt-get install llvm-9
LLVM_CONFIG=/usr/bin/llvm-config-9 pip3 install keras librosa pandas sklearn matplotlib

# installing tensorflow (https://qengineering.eu/install-tensorflow-2.2.0-on-raspberry-pi-4.html first yellow block)
sudo apt-get install gfortran libhdf5-dev libc-ares-dev libeigen3-dev libatlas-base-dev libopenblas-dev libblas-dev liblapack-dev cython
pip3 install gdown
gdown https://drive.google.com/uc?id=11mujzVaFqa7R1_lB7q0kVPW22Ol51MPg
pip3 install tensorflow-2.2.0-cp37-cp37m-linux_armv7l.whl
```

## Running the script
To run the script, the following steps can be taken:
1) `cd` into the cloned git repository on the Raspberry Pi: `cd ~/Documents/esa/experiments/audio-sentiment-analysis`. 
2) activate the Python Virtual Environment in this folder by running `source bin/activate` 
3) run the script by executing the command `python3 src/us15_model_to_run_in_raspberry.py`

When done correctly, the script should give an output similar to the following:
```
Loaded model from disk
         0          1          2          3          4          5    ...        210        211        212        213        214        215
           0          0          0          0          0          0  ...          0          0          0          0          0          0
0 -18.203562 -21.471832 -22.522213 -21.712259 -22.264282 -20.707907  ... -25.210173 -24.740648 -22.311916 -22.579805 -22.314659 -21.552433

[1 rows x 216 columns]
1/1 [==============================] - 0s 735us/step
[[0.9855444  0.01445557]]
['Angry']
```
