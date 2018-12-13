import glob
import json

config_jsons = []

for f_name in glob.iglob("/home/cleptes/Programming/Python/ml_metal_genre_classification/only_best_configs/*.json"):
    config_jsons.append(json.load(open(f_name)))

print(len(config_jsons))

sorted_configs = sorted(config_jsons,key=lambda k: k["accuracy"], reverse=True)

print(json.dumps(sorted_configs[:3],indent=2))


print("model_name,accuracy,epoch,no_nodes1,no_nodes2")

nnodes_1024_1024 = []
nnodes_1024_512 = []

for conf in sorted_configs:
    print("%s,%f,%d,%d,%d" % (conf["model_name"],conf["accuracy"],conf["no_of_epochs"],conf["n_nodes"][0],conf["n_nodes"][1]))
    if (conf["n_nodes"][1]==512):
        nnodes_1024_512.append((conf["no_of_epochs"],conf["accuracy"]))
    else:
        nnodes_1024_1024.append((conf["no_of_epochs"],conf["accuracy"]))

x_512 = [x[0] for x in sorted(nnodes_1024_512)]
y_512 = [x[1] for x in sorted(nnodes_1024_512)]

x_1024 = [x[0] for x in sorted(nnodes_1024_1024)]
y_1024 = [x[1] for x in sorted(nnodes_1024_1024)]
import matplotlib.pyplot as plt


print(x_512)
print(y_512)
plt.plot(x_512,y_512, color="green")
plt.plot(x_1024,y_1024, color="red")
plt.show()