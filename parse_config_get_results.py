import glob
import json

config_jsons = []

for f_name in glob.iglob("/home/cleptes/Programming/Python/ml_metal_genre_classification/model_configs/*.json"):
    config_jsons.append(json.load(open(f_name)))

print(len(config_jsons))

sorted_configs = sorted(config_jsons,key=lambda k: k["accuracy"], reverse=True)
print("BEST:\n",json.dumps(sorted_configs[0],indent=2, sort_keys=True))
best_accurs = []

for conf in sorted_configs:
    if(conf["accuracy"] > 0.749999 or True):
        best_accurs.append(conf)
        #print("name: %s, accuracy: %f" % (conf["model_name"],conf["accuracy"]))
    else:
        break

best_1_layer = []
best_2_layer = []
best_3_layer = []

for best in best_accurs:
    try:
        if best["n_layers"] == 1:
            best_1_layer.append(best)
        elif best["n_layers"] == 2:
            best_2_layer.append(best)
        else:
            best_3_layer.append(best)
    except:
        best_1_layer.append(best)

d_3_l = {}

for i in range(len(best_3_layer)):
    try:
        d_3_l[best_3_layer[i]["type_of_data"]] = {}
    except:        
        d_3_l["16k_512fft_padding"] = {}
        best_3_layer[i]["type_of_data"] = "16k_512fft_padding"


for i in range(len(best_2_layer)):
    try:
        d_3_l[best_2_layer[i]["type_of_data"]] = {}
    except:        
        d_3_l["16k_512fft_padding"] = {}
        best_2_layer[i]["type_of_data"] = "16k_512fft_padding"

for i in range(len(best_1_layer)):
    best_1_layer[i]["n_nodes"]=1024
    try:
        d_3_l[best_1_layer[i]["type_of_data"]] = {}
    except:        
        d_3_l["16k_512fft_padding"] = {}
        best_1_layer[i]["type_of_data"] = "16k_512fft_padding"


for x in best_3_layer:
    print(x["type_of_data"]+","+str(x["n_nodes"])+","+str(x["no_of_epochs"])+","+str(x["accuracy"]))

print("2_layers")

for x in best_2_layer:
    print(x["type_of_data"]+","+str(x["n_nodes"])+","+str(x["no_of_epochs"])+","+str(x["accuracy"]))

print("1_layers")

for x in best_1_layer:
    print(x["type_of_data"]+","+str(x["n_nodes"])+","+str(x["no_of_epochs"])+","+str(x["accuracy"]))

'''    try:
        d_3_l[x["type_of_data"]][tuple(x["n_nodes"])].append((x["no_of_epochs"],x["accuracy"]))
        d_3_l[x["type_of_data"]][tuple(x["n_nodes"])] = sorted(d_3_l[x["type_of_data"]][tuple(x["n_nodes"])])
    except:
        d_3_l[x["type_of_data"]][tuple(x["n_nodes"])]=[(x["no_of_epochs"],x["accuracy"])]
    #print(x["n_nodes"]) 
print(d_3_l.keys())

to_plot_3_3 = {}
for k, v in d_3_l.items():
    to_plot_3 = []
    for k, a in v.items():
        x = [i[0] for i in a]
        print(x)
        y = [i[1] for i in a]
        to_plot_3.append((k,x,y))
    to_plot_3_3[k] = to_plot_3



import matplotlib.pyplot as plt

for k, to_plot_3 in to_plot_3_3.items():
    for to_plot in to_plot_3:
        print(to_plot[1])
        plt.plot(to_plot[1], to_plot[2], label=to_plot[0])
    plt.show()


'''

d_2_l = {}
for x in best_2_layer:
    try:
        d_2_l[tuple(x["n_nodes"])].append((x["no_of_epochs"],x["accuracy"]))
        sorted(d_2_l[tuple(x["n_nodes"])])
    except:
        d_2_l[tuple(x["n_nodes"])]=[(x["no_of_epochs"],x["accuracy"])]
print(d_2_l.keys())

a_1_l = []
for x in best_1_layer:
    a_1_l.append((x["no_of_epochs"],x["accuracy"]))
    


'''
print("1 layer",len(best_1_layer))
print("2 layer",len(best_2_layer))
print("3 layer",len(best_3_layer))

# 2 layer is best print details:
epoch_times_1024_512 = []
acc_1024_512 = []
epoch_times_1024_1024 = []
acc_1024_1024 = []
for best in best_2_layer:
    if [1024,512] == best["n_nodes"]:
        epoch_times_1024_512.append(best["no_of_epochs"])
        acc_1024_512.append(best["accuracy"])
    else:
        epoch_times_1024_1024.append(best["no_of_epochs"])
        acc_1024_1024.append(best["accuracy"])
    print("n_nodes:",best["n_nodes"],", acc:", best["accuracy"], "no_epochs:",best["no_of_epochs"], "data:",best["type_of_data"])

avg_epoc_1024_512 = sum(epoch_times_1024_512)/len(epoch_times_1024_512)
print("For 1024,512 nodes avg epoch:",avg_epoc_1024_512)
print("For 1024,512 nodes avg acc:",sum(acc_1024_512)/len(acc_1024_512))
avg_epoc_1024_1024 = sum(epoch_times_1024_1024)/len(epoch_times_1024_1024)
print("For 1024,1024 nodes avg epoch:",avg_epoc_1024_1024)
print("For 1024,1024 nodes avg acc:",sum(acc_1024_1024)/len(acc_1024_1024))

config_to_run = [
    {
        "data": "32k_1024fft_padding",
        "epochs": [i for i in range(int(avg_epoc_1024_512-5),int(avg_epoc_1024_512+6))],
        "no_nodes": (1024,512),
        "no_of_layers": 2,
    },
    {
        "data": "32k_1024fft_padding",
        "epochs": [i for i in range(int(avg_epoc_1024_1024-5),int(avg_epoc_1024_1024+6))],
        "no_nodes": (1024,1024),
        "no_of_layers": 2,
    }
]

print(json.dumps(config_to_run, indent=2))
json.dump(config_to_run, open("config_to_run.json", "w"),indent=2)
#print(json.dumps(sorted_configs[-1],indent=2,sort_keys=True))
'''