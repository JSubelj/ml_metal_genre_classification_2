# -*- coding: utf-8 -*-


# correct all dirs so that they are valid
# run dataset_tools to create dataset
#
#
#
#
#
#
#
import random
import string

no_of_samples_per_split = 1000
no_of_coef_per_sample = 13

size_of_single_pickle = 0.100  # aprox. in MB
ram_to_use = 2000  # in MB
batch_size = int(ram_to_use / size_of_single_pickle)
no_of_epochs = 20

train_percentage = 0.7
test_percentage = 0.25
validate_percentage = 0.05

slice_by_genre_name = "%s_*_*.pckl"
slice_name = "%s_%d_%d.pckl"
root_directory = "/home/cleptes/Programming/Python/ml_metal_genre_classification/"
pickles_directory = "/home/cleptes/Programming/Python/genre_classification_data/splits/"
dataset_directory = root_directory + "datasets/"
model_directory = root_directory + "model/"

models_directory = "/mnt/win/ml_metal_genre_models/"
models_config_directory = root_directory + "model_configs/"



rand_string_f = lambda: ''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(20))
rand_string = rand_string_f()
def init_model_names(rand_string):
    model_name_tweaker_template = "mtl_clsf_DNN_nepoch_%d_nlayers_%d_%s"+rand_string
    model_name_tweaker = model_name_tweaker_template+".tflearn"
    config_name_tweaker = model_name_tweaker_template+".json"
    return model_name_tweaker, config_name_tweaker
model_name_tweaker, config_name_tweaker = init_model_names(rand_string)

intermed_tweaking_directory = root_directory + "intermed_tweak/"
intermed_tweaking_config_name = "intermed_cnfg.json"

intermed_model_directory = root_directory + "intermed_model/"
log_file_training = root_directory + "logs/train.log"
log_file_validating = root_directory + "logs/validate.log"
log_file_ds_creation = root_directory + "logs/ds_creation.log"

model_name = "metal_classification_DNN.tflearn"
dataset_name = "dataset_%s.pckl"
genres = ["heavy", "death", "thrash", "black"]
