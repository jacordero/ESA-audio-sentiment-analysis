# Testing
To verify the correctness of the different software modules within the STERN audio module, this repository contains testing code. This testing code is mostly `TODO system tests?` implemented using PyTest 6.1.1. It consists of several test cases, which check different aspects of the STERN audio module. The test cases are split into the following categories:

- **Model Interface test cases**, which test whether the interfaces of the deep learning model used by the system conform to the specification
- **Threshold test cases**, which test whether the performance and quality metrics of the deep learning model used by the system are above a set threshold
- **Preprocessing test cases**, which check the parts of the STERN audio module which process audial input so it can be fed into the deep learning model.
- **Postprocessing test cases**, which check the logging functionality of the STERN audio module.

The structure of these tests is described under [Test structure](#test-structure). The test cases. The procedure to execute these test cases can be found at [Test procedure](#test-procedure). 

Most tests have been included in a Gitlab CI/CD pipeline has also been used throughout the development of STERN. The section on [Testing pipeline](#testing-pipeline) describes how tests are integrated in this pipeline. 

`TODO system tests?` 

## Test structure

## Test procedure


## Testing Pipeline