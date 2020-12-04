# Testing pipeline

In order to automatically assess the models that AI engineers produce, we decided to create an automated pipeline, which will assess several properties and concerns. The pipeline consists of several stages. In every stage, we assess a specific set of properties. The current stages of the pipeline are:

  - PEP8 Conformity
  - Test Data Verification
  - Model Architecture Verification
  - Threshold Verification
  - Smoke Test

# Instructions
When an engineer develops a new candidate model, then the pipeline takes over the assessment of the model. The engineer has to upload the trained model to DVC, which is where the models and data are stored. Then, the .dvc files of the project change their hash values. These files are stored in GitLab. As a result, the engineer needs to push the changes to GitLab. Once the changes to the .dvc files are pushed, the pipeline takes over and the stages of the pipeline are executed sequentially. For the stages to run we employ two different GitLab runners. The first runner runs on TU/e's servers and it is a docker runner. The second runner runs on the Raspberry Pi. The outcome of the pipeline is twofold. First, a graphical representation of the results can be found on the GitLab web interface, which the engineer can use to verify the results. Also, a JSON file is generated in the Raspberry Pi, which contains all the results of the performance assessment of the candidate model. 

Below you can find a small description for each stage of the pipeline.

# Pep8 Conformity
In this stage, we are assessing the quality of the code. We created a docker image that contains the latest code, which is then assessed.

# Test Data Verification
In this stage, we are assessing the quality of data against certain assumptions that we made as a team.

# Model Architecture Verification
In this stage, we are assessing the interface of the candidate model. The model shall always conform to the specified architecture (input shape and output shape).

The test cases that run in this stage are assessing the architecture of the candidate model.

# Threshold Verification
In this stage, we are assessing the performance of the candidate model. We created a generic script that gets as an input a candidate model. The directory, filename, and type of the candidate model are specified in a configuration file. Then, the generic script computes a set of performance metrics and writes them into a JSON file. Finally, the candidate's performance is assessed against the deployed model's performance and general "goals" that the architect introduced on each of the performance metrics.

The generic script is located in the src folder and it's called compoute_performance.py. In order to manually run this script you first need to specify the correct candidate model in the configuration.yml, which can be located in the same folder. Furthermore, the candidate model and the test data must also be tracked by the DVC.

# Smoke Test
In this stage, we are running the complete system for two iterations, such that we are positive about the most important functions of the system.

# Dependencies
The new dependencies that the testing pipeline introduces are:
  - pytest, 5.4.3
  - pytest-pep8

# Description of the configuration file

The testing pipeline is based on the configuration.yml file. It contains several fields, which are used for testing and assessment purposes. The fields *sequencial_model* and *siamese_model* contain the expected input and output shape of the respective models. These fields are used in order to verify that the model interfaces correctly with the native components. The last field is *target_metrics*. It contains values, such as accuracy, performance, f1 score, confidence, and per class accuracy. These values describe the hard requirements for the behavior of the model, which are specified by the architect. 



