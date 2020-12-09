# Testing
To verify the correctness of the different software modules within the STERN audio module, this repository contains testing code. This testing code is mostly implemented using PyTest 6.1.1. It consists of several test cases, which check different aspects of the STERN audio module. The test cases are split into the following categories:

- **Model Interface test cases**, which test whether the interfaces of the deep learning model used by the system conform to the specification
- **Threshold test cases**, which test whether the performance and quality metrics of the deep learning model used by the system are above a set threshold
- **Preprocessing test cases**, which check the parts of the STERN audio module which process audial input so it can be fed into the deep learning model.
- **Postprocessing test cases**, which check the logging functionality of the STERN audio module.
- **Data test cases**, which check whether the datasets used during training and testing are of the correct format.
- **System tests**, which test the complete system functionality.

The structure of these tests is described under [Test structure](#test-structure). The test cases. The procedure to execute these test cases can be found at [Test procedure](#test-procedure). A complete definition of all test cases can be found in the Software Transfer Document.

## Test structure

All testing code is located in the [tests](/tests/) folder of this repository. A complete overview of the files in this folder is show below.

```
tests
├─compute_models_performance.py
├─configuration.yml
├─context.py
├─data_loader.py
└───test_cases
    ├─Model_test_cases
    ├─Postprocessing
    ├─Preprocessing
    ├─Data
    └─Threshold_test_cases
```

The [test_cases](/tests/test_cases/) folder contains the code for all the test cases. There are multiple folders inside of the [test_cases](/tests/test_cases/) folder, representing the different test case categories.

Some test-related files are also located directly inside the [tests](/tests/)  folder. These are not related to any specific test case, but they are used in the testing process. 

Note that there is no code available relating to the System tests. Executing the System tests involves running the STERN audio module itself, which is located in the [src](/src/) folder.

## Test procedure

Test cases can be run using PyTest. The following command can be used from the root of the repository to run all the test cases:

```
pytest tests/test_cases/ --disable-warnings
```

To run the System tests, one has to run the STERN audio module itself. instructions on how to do that can be found in the [main readme file](/STERN_Audio_README.md). The `input_type` option found in [raspi_candidate_config.yml](/src/raspi_candidate_config.yml) can be used to configure which System test is run.