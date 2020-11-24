import os
import yaml
import shutil

def copy_model(origin_dir, destination_dir, model_dir):

    origin = os.path.join(origin_dir, model_dir)
    destination = os.path.join(destination_dir, model_dir)
    print("Deployed directory before: {}".format(os.listdir(destination_dir)))    
    shutil.copytree(origin, destination)
    print("Deployed directory after: {}".format(os.listdir(destination_dir)))

if __name__ == "__main__":

    prod_config_file = "raspi_deployment_config.yml"
    candidate_config_file = "raspi_candidate_config.yml"

    with open(prod_config_file) as input_file:
        prod_config_parameters = yaml.load(input_file, Loader=yaml.FullLoader)

    with open(candidate_config_file) as input_file:
        candidate_config_parameters = yaml.load(
            input_file, Loader=yaml.FullLoader)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = candidate_config_parameters['models']['tone_model']['dir'].split(
        '/')[0]
    prod_models_dir = prod_config_parameters['prod_models_dir']
    prod_models_path = os.path.normpath(os.path.join(script_dir, prod_models_dir))
    candidate_models_dir = candidate_config_parameters['prod_models_dir']
    candidate_models_path = os.path.normpath(os.path.join(script_dir, candidate_models_dir))

    copy_model(candidate_models_path, prod_models_path, model_dir)

