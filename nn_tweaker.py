from time import time
import train_NN
import dataset_tools
import sys
import json
import test_NN
import os


def save_config(cnfg):
    json.dump(cnfg, open(config.intermed_tweaking_directory + config.intermed_tweaking_config_name, "w"))


def get_number_for_each_genre(x):
    retdict = {}
    for genre in config.genres:
        retdict[genre] = 0

    for fname in x:
        for genre in config.genres:
            if genre in fname:
                retdict[genre] += 1

    return retdict, len(x)


def get_procentage(gen_dict, n_all_files):
    retdict = {}
    for genre in config.genres:
        retdict[genre] = gen_dict[genre] / n_all_files

    return retdict, n_all_files


def load_config():
    return json.load(open(config.intermed_tweaking_directory + config.intermed_tweaking_config_name, "r"))


def remove_intermed_config():
    folder = config.intermed_tweaking_directory[:-1]
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def save_model(model, epoch_time, n_layers, time_passed):
    model.save(config.models_directory + config.model_name_tweaker % (epoch_time, n_layers, type_of_data))
    json.dump(
        {"model_name": config.model_name_tweaker % (epoch_time, n_layers,type_of_data), "no_of_epochs": epoch_time,
         "time_passed_training": time_passed},
        open(config.models_config_directory + config.config_name_tweaker % (epoch_time, n_layers,type_of_data), "w"))
        


def save_details(epoch_time, time_passed, acc, miniacc, len_train, len_test, gendict_train, gendict_test, proc_train,
                 proc_test, n_layers, n_nodes):
    details = json.load(open(config.models_config_directory + config.config_name_tweaker % (epoch_time, n_layers,type_of_data), "r"))
    details["time_passed_testing"] = time_passed
    details["accuracy"] = acc
    details["mini_accs"] = miniacc
    details["no_train_files"] = len_train
    details["no_test_files"] = len_test
    details["gendict_train"] = gendict_train
    details["gendict_test"] = gendict_test
    details["proc_train"] = proc_train
    details["proc_test"] = proc_test
    details["n_layers"] = n_layers
    details["n_nodes"] = n_nodes
    details["type_of_data"] = type_of_data
    json.dump(details, open(config.models_config_directory + config.config_name_tweaker % (epoch_time, n_layers,type_of_data), "w"))


def tweak_by_epochtime(epoch_from, epoch_time_decrement=3, epoch_to=0, n_layers=1, n_nodes=(1024,), load=False):
    assert len(n_nodes) == n_layers

    start_time = time()
    no_of_epochs = epoch_from

    if load:
        no_of_epochs = load_config()["no_of_epochs"]

    X, y = dataset_tools.import_dataset("train")
    X_test, y_test = dataset_tools.import_dataset("test")

    gendict_train, n_all_train = get_number_for_each_genre(dataset_tools.import_dataset("train")[0])
    print("train", gendict_train)
    proc_train = get_procentage(gendict_train, n_all_train)
    print("train", proc_train)
    gendict_test, n_all_test = get_number_for_each_genre(dataset_tools.import_dataset("test")[0])
    print("test", gendict_test)
    proc_test = get_procentage(gendict_test, n_all_test)
    print("test", proc_test)

    for i in range(no_of_epochs, epoch_to - 1, -epoch_time_decrement):
        start_time_ = time()
        print("training model with", i, "epoch time")
        save_config({"model_name": "intermed_model", "no_of_epochs": i, "n_layers": n_layers, "n_nodes": n_nodes})
        model = train_NN.train_NN(X, y, n_epochs=i, continue_work=load, n_layers=n_layers, n_nodes=n_nodes)
        time_passed = time() - start_time_
        print("trained! time:")

        save_model(model, i, n_layers, time_passed)

        start_time_ = time()
        print("testing model with", i, "epoch time")
        acc, miniacc = test_NN.test_NN(model, X_test, y_test)
        print("Done! acc:", acc, "time:", time() - start_time_, "s")
        save_details(i, time() - start_time_, acc, miniacc, n_all_train, n_all_test, gendict_train, gendict_test,
                     proc_train, proc_test, n_layers, n_nodes)
        remove_intermed_config()

    print("Finished all! time:", time() - start_time)


def run_tweaker_from_config(config_name="/home/cleptes/Programming/Python/ml_metal_genre_classification/config_to_run.json", config_store_dir="/home/cleptes/Programming/Python/only_best_configs/"):
    # TODO: prekopiri v main
    import config
    import dataset_tools
    config_to_run = json.load(open(config_name))
    config.models_config_directory = config_store_dir
    start = time()
    for cnf in config_to_run:
        for epoch in cnf["epochs"]:
            type_of_data = cnf["data"]
            dataset_tools.create_dataset_from_slices()
            config.model_name_tweaker, config.config_name_tweaker = config.init_model_names(config.rand_string_f())
            tweak_by_epochtime(epoch, epoch_to=epoch, n_layers=cnf["no_of_layers"], n_nodes=tuple(cnf["no_nodes"]), load=False)
    print("End: ",time()-start)
    #for conf in config_to_run:


    #dataset_tools.create_dataset_from_slices()




if __name__ == "__main__":
    #run_tweaker_from_config()
    config_name="/home/cleptes/Programming/Python/ml_metal_genre_classification/config_to_run.json"
    config_store_dir="/home/cleptes/Programming/Python/ml_metal_genre_classification/only_best_configs/"
    import config
    import dataset_tools
    config_to_run = json.load(open(config_name))
    config.models_config_directory = config_store_dir
    start = time()
    for cnf in config_to_run:
        for epoch in cnf["epochs"]:
            type_of_data = cnf["data"]
            dataset_tools.create_dataset_from_slices()
            config.model_name_tweaker, config.config_name_tweaker = config.init_model_names(config.rand_string_f())
            tweak_by_epochtime(epoch, epoch_to=epoch, n_layers=cnf["no_of_layers"], n_nodes=tuple(cnf["no_nodes"]), load=False)
    print("End: ",time()-start)

    
    '''gendict, n_all = get_number_for_each_genre(dataset_tools.import_dataset("train")[0])
    print("train", gendict)
    print("train", get_procentage(gendict, n_all))
    gendict, n_all = get_number_for_each_genre(dataset_tools.import_dataset("test")[0])
    print("test", gendict)
    print("test", get_procentage(gendict, n_all))  
    from importlib import reload
    import config
    start = time()
    type_of_data = sys.argv[1]
    import dataset_tools

    dataset_tools.create_dataset_from_slices()
    tweak_by_epochtime(epoch_from=config.no_of_epochs)
    config = reload(config)
    dataset_tools.create_dataset_from_slices()
    tweak_by_epochtime(n_layers=2, n_nodes=(1024, 1024), epoch_from=config.no_of_epochs)
    config = reload(config)
    dataset_tools.create_dataset_from_slices()
    tweak_by_epochtime(n_layers=2, n_nodes=(1024, 512), epoch_from=config.no_of_epochs)
    config = reload(config)
    dataset_tools.create_dataset_from_slices()
    tweak_by_epochtime(n_layers=3, n_nodes=(1024, 512, 256), epoch_from=config.no_of_epochs)
    config = reload(config)
    dataset_tools.create_dataset_from_slices()
    tweak_by_epochtime(n_layers=3, n_nodes=(1024, 1024, 512), epoch_from=config.no_of_epochs)
    config = reload(config)
    dataset_tools.create_dataset_from_slices()
    tweak_by_epochtime(n_layers=3, n_nodes=(1024, 1024, 1024), epoch_from=config.no_of_epochs)
    print("full run: ",time()-start,"s")
    '''
