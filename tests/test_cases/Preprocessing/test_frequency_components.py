import yaml

# load configuration files
def load_configuration(filename):
    prod_config_file = filename
    with open(prod_config_file) as input_file:
        config_parameters = yaml.load(input_file, Loader=yaml.FullLoader)
    return config_parameters

def load_both_configurations():
    candidate = load_configuration("src/raspi_candidate_config.yml")
    deployed = load_configuration("src/raspi_deployed_config.yml")
    return candidate, deployed

def test_1a_channels():
    candidate, deployed = load_both_configurations()
    assert candidate["audio_channels"] > 0
    assert deployed["audio_channels"] > 0

def test_1b_frequency():
    candidate, deployed = load_both_configurations()
    assert candidate["audio_frequency"] > 0
    assert deployed["audio_frequency"] > 0