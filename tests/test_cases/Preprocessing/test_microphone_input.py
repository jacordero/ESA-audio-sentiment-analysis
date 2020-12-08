import yaml

# load configuration files
def load_configuration(filename):
    prod_config_file = filename
    with open(prod_config_file) as input_file:
        config_parameters = yaml.load(input_file, Loader=yaml.FullLoader)
    return config_parameters

def load_both_configurations():
    candidate = load_configuration("src/raspi_candidate_config.yml")
    deployed = load_configuration("src/raspi_deployment_config.yml")
    return candidate, deployed

def test_microphone_input_2():
    candidate, deployed = load_both_configurations()
    assert candidate["audio_length"] > 0
    assert deployed["audio_length"] > 0