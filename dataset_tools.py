from config import genres, pickles_directory, slice_by_genre_name, no_of_coef_per_sample, no_of_samples_per_split
import config
import os
import glob
import random
import numpy as np
import pickle
from time import time



def save_dataset(train_X, train_y, test_X, test_y, validation_X, validation_y):
    pickle.dump((train_X, train_y), open(config.dataset_directory+config.dataset_name % "train", "wb"))
    pickle.dump((test_X, test_y), open(config.dataset_directory+config.dataset_name % "test", "wb"))
    pickle.dump((validation_X, validation_y), open(config.dataset_directory+config.dataset_name % "validate", "wb"))

def import_dataset(mode):
    assert mode == "train" or mode == "test" or mode == "validate"
    if mode == "train":
        train_X, train_y = pickle.load(open(config.dataset_directory+config.dataset_name % "train", "rb"))
        return train_X, train_y

    if mode == "test":
        test_X, test_y = pickle.load(open(config.dataset_directory+config.dataset_name % "test", "rb"))
        return test_X, test_y

    if mode == "validate":
        validation_X, validation_y = pickle.load(open(config.dataset_directory+config.dataset_name % "validate", "rb"))
        return validation_X, validation_y


def create_dataset_from_slices(validation_ratio=config.validate_percentage, test_ration=config.test_percentage, slices_per_genre=None):
    assert validation_ratio + test_ration < 0.99
    print("started")
    start_time = time()
    data = []

    for genre in genres:
        print("genre:",genre)
        file_names = [file for file in glob.glob(pickles_directory+slice_by_genre_name % genre)]
        file_names = file_names[:slices_per_genre]

        random.shuffle(file_names)
        print(len(file_names), "of files for genre", genre)
        # label genre array len(label) == len(genre) 1 for genre 0 for others
        for file_name in file_names:
            #slice_data = pickle.load(open(file_name, "rb"))
            label = [1. if genre == g else 0. for g in genres]
            data.append((file_name,label))

    random.shuffle(data)

    X,y = zip(*data)

    no_for_validation = int(len(X)*validation_ratio)
    no_for_test = int(len(X)*test_ration)
    no_for_train = len(X) - (no_for_validation+no_for_test)

    #train_X = np.array(X[:no_for_train]).reshape([-1, no_of_samples_per_split, no_of_coef_per_sample, 1])
    train_X = X[:no_for_train]
    train_y = np.array(y[:no_for_train])
    #validation_X = np.array(X[no_for_train:no_for_train+no_for_validation]).reshape([-1, no_of_samples_per_split, no_of_coef_per_sample, 1])
    validation_X = X[no_for_train:no_for_train+no_for_validation]
    validation_y = np.array(y[no_for_train:no_for_train+no_for_validation])
    #test_X = np.array(X[-no_for_test:]).reshape([-1, no_of_samples_per_split, no_of_coef_per_sample, 1])
    test_X = X[-no_for_test:]
    test_y = np.array(y[-no_for_test:])

    save_dataset(train_X, train_y, test_X, test_y, validation_X, validation_y)

    print(len(train_y), "for train,", len(test_y), "for test,",len(validation_y),"for validation")
    with open(config.log_file_ds_creation, "a") as f:
        f.write("---------------------------------------------------------------------------------------------------\n")
        f.write("files: validation - " + str(config.validate_percentage * 100) + "%, test - " + str(
            config.test_percentage * 100) + "%, train - " + str(config.train_percentage * 100) + "% " + "\n")
        f.write(str(len(train_y))+" files for training, "+str(len(test_y))+" files for testing, "+str(len(validation_y))+" files for validation.\n")
        f.write("summas summarum time: "+str(time()-start_time)+"s\n")

    return train_X, train_y, test_X, test_y, validation_X, validation_y


if __name__=="__main__":
    create_dataset_from_slices()