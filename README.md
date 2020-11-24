## Audio Sentiment Analysis

Repository for the audio sentiment analysis module of the ESA project.

## Setup the project

1. Clone this project:
```
git clone https://ooti-projects.win.tue.nl/gitlab/st-c2019/esa/audio-sentiment-analysis.git
```

2. Follow the instructions to install DVC (on Windows): 
* Linux: https://dvc.org/doc/install/linux
* MacOS: https://dvc.org/doc/install/macos
* Windows: https://dvc.org/doc/install/windows 

3. We use a dedicated Raspberry Pi to store models and data. We connect to this Raspberry Pi using ssh commands. To enable this connection, request access to the VPN that manages the Raspberry Pis.

4. Create and activate the virtual environment that will be used to executed cloned repository.

5. Start the Raspberry Pi VPN and run the following command to download production data and models:
```
> dvc pull
```

## Use cases

**Note:** before performing the following use cases, always pull the latest version of the production models and datasets (```dvc pull```).

### Modify production data
1. Add or remove audios from the ```prod_data``` directory.
2. Execute ```dvc status``` to see that the contents of ```prod_data``` were modified.
3. Execute ```dvc add prod_data``` to update the contents of ```prod_data.dvc```.
4. Track changes with git: ```git add prod_data.dvc```.
5. Git commit: ```git commit -m "commit message" ```.
6. Update production: ```dvc push```.
7. Push changes to GitLab: ```git push origin [branch-name]```.

### Add new production model
1. Add new model in the ```prod_models``` directory.
2. Execute ```dvc status``` to see that the contents of ```prod_models``` were modified.
3. Execute ```dvc add prod_models``` to update the contents of ```prod_models.dvc```.
4. Track changes with git: ```git add prod_models.dvc```.
5. Git commit: ```git commit -m "commit message" ```.
6. Update production: ```dvc push```.
7. Push changes to GitLab: ```git push origin [branch-name]```.


### Deploy new model in the raspberry pi
1. Log into the raspberry pi.
2. Go to **/home/pi/Documents/esa**.
3. Activate virtual environment: ``` source venv/bin/activate ```.
4. Go to the repository containing the source code.
4. Pull changes from master ```git pull origin master```.
5. Pull latest data and models using dvc ```dvc pull```.
6. Run stern audio script ```python src/stern_audio.py [configuration_filename]```.

The ```configuration_filename``` variable should be one of the following names:
* raspi_candidate_config.yml
* raspi_deployment_config.yml

These configuration files are available inside the src directory.

**Note:** It is recommended to always use the raspi_candidate_config.yml file testing the code locally. The raspi_deployment_config.yml is used by the testing pipeline to automatically update the candidate models uploaded to production.
