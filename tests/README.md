# Testing pipeline

In order to automatically assess the models that AI engineers produce, we decided to create an automated pipeline, which will assess several properties and concers. The pipeline consists of several stages. In every stage we assess a specific set of properties. Below you can find the properties that are assessed in each stage.
 

  - PEP8 Conformity
  - Test Data Verification
  - Model Architecture Verification
  - Threshold Verification
  - Smoke Test


# Pep8 Conformity
In this stage, we are assessing the quality of the code. We created a docker image that contains the latests code, which is then assessed.

# Test Data Verification
In this stage, we are assessing the quality of data against certain assumptions that we made as a team.

# Model Architecture Verification
In this stage, we are assessing the interface of the candidate model. The model shall always conform to the specifed architecture (input shate and output shape).

# Threshold Verification
In this stage, we are assessing the performance of the candidate model. We created a generic script that gets as an input a candidate model. The directory, filename, and type of the candidate model are specified in a configuration file. Then, the generic script computes a set of performance metrics and writes them into a json file. Finally, the candidate's performance is assessed against the deployed models performance and general "goals" that the architect introduced on each of the performance metrics.

# Smoke Test
In this stage, we are running the complete system for two iterations, such that we are positive about the most important functions of the system.
