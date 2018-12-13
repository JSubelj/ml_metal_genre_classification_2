from init_model import create_model
import dataset_tools
import random
import string
import config
import time
import numpy as np
import pickle
import tflearn
import sys
import tensorflow as tf
import os

def pickle_lazy_loading(Xs, ys, batch_size=config.batch_size, i=0):
    len_of_whole = len(ys)
    proccessed_till_now = 0
    while i * batch_size < len_of_whole:
        if (i + 1) * batch_size < len_of_whole:
            print("processing from", i * batch_size, "to", (i + 1) * batch_size, "files")
        else:
            print("processing from", i * batch_size, "to", len_of_whole, "files")
        print("proccessed till now:", proccessed_till_now)
        ys_splice = ys[i * batch_size:(i + 1) * batch_size]
        Xs_splice = np.array(
            [pickle.load(open(file_name, "rb")) for file_name in Xs[i * batch_size:(i + 1) * batch_size]]).reshape(
            [-1, config.no_of_samples_per_split, config.no_of_coef_per_sample, 1])
        proccessed_till_now += len(ys_splice)

        yield Xs_splice, ys_splice
        i += 1


def save_batch(model, no_of_batch_done):
    model.save(config.intermed_model_directory + config.model_name)
    with open(config.intermed_model_directory + "no_of_batch_done.txt", "w") as f:
        f.write(str(no_of_batch_done))


def load_batch(model):
    try:
        with open(config.intermed_model_directory + "no_of_batch_done.txt", "r") as f:
            x = int(f.read())
        model.load(config.intermed_model_directory + config.model_name)
    except:
        x = 0
    return model, x

def remove_batch():
    folder = config.intermed_model_directory[:-1]
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def train_NN(train_X, train_y, n_epochs=config.no_of_epochs, continue_work=False, n_layers=1, n_nodes=(1024,)):
    tf.reset_default_graph()
    model = create_model(no_of_layers=n_layers,num_of_nodes=n_nodes)

    i = 0
    iterator_batch = 0
    if continue_work:
        model, iterator_batch = load_batch(model)

    tflearn.init_graph(seed=1995, gpu_memory_fraction=1)
    with tf.Session() as sess:
        tflearn.is_training(True, sess)
    for train_batch_X, train_batch_y in pickle_lazy_loading(train_X, train_y, i=iterator_batch):
        print("training batch:", i)
        start_time__ = time.time()
        model.fit(train_batch_X, train_batch_y, n_epoch=n_epochs, shuffle=True, snapshot_step=100,
                  show_metric=True)
        print("batch", i, "trained in", time.time() - start_time__, "s")
        i += 1
        save_batch(model, i)

    remove_batch()

    return model


if __name__ == "__main__":
    '''
    load = True
    if len(sys.argv) == 2 and sys.argv[1] == "-l":
        pass
    elif len(sys.argv) == 2 and sys.argv[1] == "-n":
        load = False
    else:
        print("Missing required argument! new: -n, load: -l")
        sys.exit(0)


    start_time = time.time()
    model = create_model()
    if load:
        model, iterator_batch = load_batch(model)
    else:
        iterator_batch = 0
    model_create_time = time.time() - start_time
    print("model creation time:", model_create_time)



    start_time_ = time.time()
    train_X, train_y = dataset_tools.import_dataset("train")

    dataset_import_time = time.time() - start_time_
    print("import time:", dataset_import_time)

    #run_id = "metal genres - " + str(config.batch_size) + " " + ''.join(
     #   random.SystemRandom().choice(string.ascii_uppercase) for _ in range(10))

    start_time_ = time.time()
    i = 0
    tflearn.init_graph(seed=1995, gpu_memory_fraction=1)
    with tf.Session() as sess:
        tflearn.is_training(True, sess)
    for train_batch_X, train_batch_y in pickle_lazy_loading(train_X, train_y, i=iterator_batch):
        print("training batch:", i)
        start_time__ = time.time()
        model.fit(train_batch_X, train_batch_y, n_epoch=config.no_of_epochs, shuffle=True, snapshot_step=100,
                  show_metric=True)
        print("batch", i, "trained in", time.time() - start_time__, "s")
        i += 1
        save_batch(model, i)


    model_training_time = time.time() - start_time_
    print("training model time:", model_training_time)
    print("model trained")
    start_time_ = time.time()
    model.save(config.model_directory + config.model_name)
    model_saving_time = time.time() - start_time_
    print("saving model time:", model_saving_time)
    summas_summarum_time = time.time() - start_time
    print("summas summarum:", summas_summarum_time)

    with open(config.log_file_training, "a") as f:
        f.write("---------------------------------------------------------------------------------------------------\n")
        f.write("files: validation - " + str(config.validate_percentage * 100) + "%, test - " + str(
            config.test_percentage * 100) + "%, train - " + str(config.train_percentage * 100) + "% " + "\n")
        f.write("model_creation_time: " + str(model_create_time) + "s\n")
        f.write("dataset import time: " + str(dataset_import_time) + "s\n")
        f.write("model training time: " + str(model_training_time) + "s\n")
        f.write("model saving time: " + str(model_saving_time) + "s\n")
        f.write("summas summarum: " + str(summas_summarum_time) + "s\n")
    '''
