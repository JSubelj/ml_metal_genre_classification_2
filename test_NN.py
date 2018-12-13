from init_model import create_model
import dataset_tools
import config
import time
import tflearn
from train_NN import pickle_lazy_loading
import tensorflow as tf


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def test_NN(model, t_X, t_y):
    tflearn.init_graph(seed=1995, gpu_memory_fraction=1)
    with tf.Session() as sess:
        tflearn.is_training(False, sess)

    mini_validations = []
    for test_X, test_y in pickle_lazy_loading(t_X, t_y):
        validation_accuracy = float(model.evaluate(test_X, test_y)[0])
        print("mini accurecy:", validation_accuracy)
        mini_validations.append(validation_accuracy)
        print("current accurecy:", mean(mini_validations))

    return mean(mini_validations), mini_validations


if __name__ == "__main__":
    '''
    start_time = time.time()
    model = create_model()
    model_creation_time = time.time()-start_time

    start_time_ = time.time()
    validate_X, validate_y = dataset_tools.import_dataset("validate")
    dataset_import_time = time.time() - start_time_

    validate_X, validate_y = pickle_lazy_loading(validate_X,validate_y,batch_size=len(validate_y)).__next__()
    print("size of:",len(validate_y))
    start_time_ = time.time()
    model.load(config.model_directory+config.model_name)
    model_load_time = time.time() - start_time_

    print("Model loaded!")

    tflearn.init_graph(seed=1995, gpu_memory_fraction=1)
    start_time_ = time.time()
    with tf.Session() as sess:
        tflearn.is_training(False, sess)
    validation_accuracy = model.evaluate(validate_X,validate_y)[0]
    model_evaluation_time = time.time() - start_time_

    print("Validation accuracy:",validation_accuracy)

    with open(config.log_file_validating, "a") as f:
        f.write("---------------------------------------------------------------------------------------------------\n")
        f.write("files: validation - " + str(config.validate_percentage * 100) + "%, test - " + str(
            config.test_percentage * 100) + "%, train - " + str(config.train_percentage * 100) + "% " + "\n")
        f.write(str(len(validate_y))+" validation files.\n")
        f.write("validation accuracy: "+str(validation_accuracy)+"\n\n")
        f.write("model_creation_time: " + str(model_creation_time) + "s\n")
        f.write("dataset import time: " + str(dataset_import_time) + "s\n")
        f.write("model load time: " + str(model_load_time) + "s\n")
        f.write("model evaluation time: " + str(model_evaluation_time) + "s\n")
        f.write("summas summarum: " + str(time.time()-start_time) + "s\n")'''
