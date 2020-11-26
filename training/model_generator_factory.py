from siamese_model_generator import SiameseModel, TwoLayerSiameseModel
from sequential_model_generator import SequentialThreeConvModulesGenerator
from sequential_model_generator import SequentialTwoConvModulesGenerator, SequentialOneConvModuleGenerator

__authors__ = "Raha Sadeghi, Jorge Cordero",
__email__ = "r.sadeghi@tue.nl; j.a.cordero.cruz@tue.nl;"
__copyright__ = "TU/e ST2019"

class ModelGeneratorFactory():
    """
    Factory Class creating required model based on the model specified in the training_parameters.yml file
    """
    def __init__(self):
        self.generators = {
            'sequential_three_conv_modules_generator': SequentialThreeConvModulesGenerator,
            'sequential_two_conv_modules_generator': SequentialTwoConvModulesGenerator,
            'sequential_one_conv_module_generator': SequentialOneConvModuleGenerator,
            'siamese_three_conv_modules_generator': SiameseModel,
            'siamese_two_conv_modules_generator': TwoLayerSiameseModel,
        }

    def get_model_generator(self, model_name):
        return self.generators[model_name]()

