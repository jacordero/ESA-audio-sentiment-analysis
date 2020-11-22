import yaml

def update_configuration(old_configuration, new_configuration):

  updated_configuration = new_configuration.copy()
  updated_configuration['test_data_dir'] = old_configuration['test_data_dir']
  updated_configuration['prod_models_dir'] = old_configuration['prod_models_dir']
  return updated_configuration


if __name__ == "__main__":

  prod_config_file = "raspi_deployment_config.yml"
  candidate_config_file = "raspi_candidate_config.yml"
  
  with open(prod_config_file) as input_file:
    prod_config_parameters = yaml.load(input_file, Loader=yaml.FullLoader)

  with open(candidate_config_file) as input_file:
    candidate_config_parameters = yaml.load(input_file, Loader=yaml.FullLoader)
    
  updated_configuration = update_configuration(prod_config_parameters, candidate_config_parameters)
  with open(prod_config_file, 'w') as output_file:
    yaml.dump(updated_configuration, output_file)

