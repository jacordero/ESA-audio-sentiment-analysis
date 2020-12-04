"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved. 
@ Authors: Raha Sadeghi r.sadeghi@tue.nl; Jorge Cordero j.a.cordero.cruz@tue.nl; Parima Mirshafiei p.mirshafiei@tue.nl;
Last modified: 01-12-2020
"""

from siamese_model_generator import SiameseModelGenerator, TwoLayerSiameseModelGenerator
from sequential_model_generator import SequentialThreeConvModulesGenerator
from sequential_model_generator import SequentialTwoConvModulesGenerator, SequentialOneConvModuleGenerator

class ModelGeneratorFactory():
    """
    Factory Class creating required model based on the model specified in the training_parameters.yml file
    """
    def __init__(self):
        self.generators = {
            'sequential_three_conv_modules_generator': SequentialThreeConvModulesGenerator,
            'sequential_two_conv_modules_generator': SequentialTwoConvModulesGenerator,
            'sequential_one_conv_module_generator': SequentialOneConvModuleGenerator,
            'siamese_three_conv_modules_generator': SiameseModelGenerator,
            'siamese_two_conv_modules_generator': TwoLayerSiameseModelGenerator,
        }

    def get_model_generator(self, model_name):
        """ Returns an instance of one of the model generators stored in the generators dictionary.

        Args:
            model_name: name of the model generator to return

        Returns:
            model generator object
        """        
        return self.generators[model_name]()

