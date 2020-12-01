# test_schema_verification.py
import os
import pytest
import yamale


def test_config_file_against_schema():
    '''The purpose of the test case is to verify that
    the configuration file adheres to the correct schema.'''
    # Get schema
    schema = yamale.make_schema('tests/test_cases/Preprocessing/configuration_schema.yml')

    # Get configuration file
    config_file  = yamale.make_data('tests/configuration.yml')

    # Validate data against the schema. Throws a ValueError if data is invalid.
    raised = True
    try:
        _ = yamale.validate(schema, config_file)
        raised = False
    finally:
        assert raised != True


def test_raspi_candidate_against_schema():
    '''The purpose of the test case is to verify that
    the rapsi_candidate_config file adheres to the correct schema.'''
    # Get schema
    schema = yamale.make_schema('tests/test_cases/Preprocessing/raspi_candidate_config_schema.yml')

    # Get candidate file
    config_file  = yamale.make_data('src/raspi_candidate_config.yml')

    # Validate data against the schema. Throws a ValueError if data is invalid.
    raised = True
    try:
        _ = yamale.validate(schema, config_file)
        raised = False
    finally:
        assert raised != True


def test_raspi_deployment_against_schema():
    '''The purpose of the test case is to verify that
    the rapsi_deployment_config file adheres to the correct schema.'''
    # Get schema
    schema = yamale.make_schema('tests/test_cases/Preprocessing/raspi_deployment_config_schema.yml')

    # Get deployment file
    config_file  = yamale.make_data('src/raspi_deployment_config.yml')

    # Validate data against the schema. Throws a ValueError if data is invalid.
    raised = True
    try:
        _ = yamale.validate(schema, config_file)
        raised = False
    finally:
        assert raised != True