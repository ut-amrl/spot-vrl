"""Contains the k-means model creation and accuracy calculations used"""
__author__= "Daniel Farkash"
__email__= "dmf248@cornell.edu"
__date__= "August 10, 2022"

import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
from itertools import product

# finds a k-means clustering chosen at the knee
def cluster_model(data):

    data=data.cpu()
    data=data.numpy()
    
    # Data used to be scaled, now it is not

    # kmeans parameters
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }

    # create a set of models 
    sse = []
    models = []
    for k in range(1, 20):

        # create and fit k-means model
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        models.append(kmeans)

    # find the knee (for the number of clusters)
    kl = KneeLocator(
        range(1, 20), sse, curve="convex", direction="decreasing"
    )
    
    return models[kl.elbow-1].labels_, kl.elbow, models[kl.elbow-1]

# Creates a k-means model clustering at the knee and finds its accuracy compared to labelled groups
# also returns the k-means model itself (used in final)
# TODO: change label typpes for accuracy check if data source is changed
def accuracy_naive_model(data, labels):

    # find the k-means model and clustering found at the knee/elbow
    clusters, elbow, model = cluster_model(data)

    label_types = ["rock", "mulch", "pebble", "speedway" ,"grass", "concrete", "brick"]
    best_acc = 0
    best_dict = {}
    label_len =  len(labels)

    # for every cluster found, see which label results in the highest accuracy for that cluster
    # naive because it allows labels to be used more than once 
    # (assumes that the correct number of clusters is found)
    for i in range(elbow):

        best_part_acc = 0
        best_part_label = ""

        # for each label
        for j in range(len(label_types)):

            new_labels = np.empty(label_len, dtype=object)

            # for each cluster
            for k in range(label_len):

                if clusters[k]== i:
                    new_labels[k] = label_types[j]
                else:
                    new_labels[k] = -1

            # find the accuracy of applying that label to that cluster
            part_acc = (np.array(new_labels) == np.array(labels)).sum()/label_len

            # if it is the best so far, then choose hat label/cluster pairing
            if part_acc > best_part_acc:

                best_part_acc = part_acc
                best_part_label = label_types[j]

        # and record it in the dict
        best_dict[i] = best_part_label
    
    # re-label using best labels
    new_labels = np.empty(label_len, dtype=object)

    for i in range(label_len):
        new_labels[i] = best_dict[clusters[i]]

    # calculate overall best accuracy with the best labels
    best_acc = (np.array(new_labels) == np.array(labels)).sum()/label_len

    print("accuracy:")
    print(best_acc)
    print(best_dict)
    return model

# same as cluster_model above, but does not return model (used for l1_only, l2_only models)
def cluster(data):
    scaler = StandardScaler()
    data=data.cpu()
    data=data.numpy()
    # scaled_features = scaler.fit_transform(data)
    scaled_features = data

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }

    sse = []
    models = []
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)
        models.append(kmeans)

    kl = KneeLocator(
        range(1, 20), sse, curve="convex", direction="decreasing"
    )
    
    return models[kl.elbow-1].labels_, kl.elbow

# does the same as accuracy_naive_model above, but does not return model (used for l1_only, l2_only models)
def accuracy_naive(data, labels):

    clusters, elbow, model = cluster_model(data)

    label_types = ["rock", "mulch", "pebble", "speedway" ,"grass", "concrete", "brick"]
    best_acc = 0
    best_dict = {}
    label_len =  len(labels)

    for i in range(elbow):
        best_part_acc = 0
        best_part_label = ""
        for j in range(len(label_types)):
            new_labels = np.empty(label_len, dtype=object)
            for k in range(label_len):
                if clusters[k]== i:
                    new_labels[k] = label_types[j]
                else:
                    new_labels[k] = -1
            part_acc = (np.array(new_labels) == np.array(labels)).sum()/label_len
            if part_acc > best_part_acc:
                best_part_acc = part_acc
                best_part_label = label_types[j]
        best_dict[i] = best_part_label
    
    new_labels = np.empty(label_len, dtype=object)
    for i in range(label_len):
        new_labels[i] = best_dict[clusters[i]]
    best_acc = (np.array(new_labels) == np.array(labels)).sum()/label_len


    print("accuracy:")
    print(best_acc)
    print(best_dict)
    return best_acc



# Calculate accuracy using best random label selection after a certain number of tries 
# (not currently used), use if above methods are too slow
def accuracy(data, labels):

    clusters, elbow = cluster(data)
    best_acc = 0
    best_dict = {}
    for iter in range(10000):
        label_types = ["rock", "mulch", "pebble", "speedway" ,"grass", "concrete", "brick"]
        dict = {}
        cats = elbow
        length = len(label_types)
        while length >0 and cats >=1:
            
            rand = random.randint(0, length-1)
            type = label_types[rand]
            label_types = label_types[0:rand] + label_types[rand+1:len(label_types)]
            
            dict[type] = cats-1
            length = length-1
            cats = cats-1
        		
        new_labels = []
        for i in range(len(labels)):
            
            if labels[i] in dict:
                new_labels.append(dict[labels[i]])
            else:
                new_labels.append(-1)
        acc = (np.array(new_labels) == np.array(clusters)).sum()/len(labels)

        if acc > best_acc:
            best_acc = acc
            best_dict = dict

    print(best_acc)
    print(best_dict)
    return best_acc


# finds the cartesian product of an array of elements
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

# finds the possible label combinations given a certain number of clusters
def get_combos(elbow):
    labels = np.array(["rock", "mulch", "pebble", "speedway" ,"grass", "concrete", "brick"])
    ar = list()
    for i in range(elbow):
        ar.append(labels)

    combos = cartesian_product(*ar) 
    return combos 

# checks all possible label combinations to get true accuracy (not naive)
# do not use unless there are very few clusters (too slow)
def accuracy_exhaustive(data, labels):

    clusters, elbow = cluster(data)

    combos = get_combos(elbow)

    best_acc = 0
    best_dict = {}
    for combo in combos:
        dict = {}
        for i in range(elbow):
            dict[i] = combo[i]
        label_len =  len(labels)
    
        new_labels = np.empty(label_len, dtype=object);
        for i in range(label_len):
            new_labels[i] = dict[clusters[i]]
        acc = (np.array(new_labels) == np.array(labels)).sum()/label_len
        if acc > best_acc:
            best_acc = acc
            best_dict = dict
    print("accuracy:")
    print(best_acc)
    print(best_dict)
    return best_acc
    

# Used for testing (not required for function)
if __name__ == '__main__':
    # features, true_labels = make_blobs(
    #     n_samples=200,
    #     centers=3,
    #     cluster_std=2.75,
    #     random_state=42
    # )
    # labels = cluster(features)

    labels = ["rock", "mulch", "pebble","rock", "mulch", "pebble","rock", "mulch", "pebble"]
    dat = [0,1,2,0,1,2,0,1,2]
    acc = accuracy_naive(dat,labels)

    # cats = 7

    # labels = np.array(["rock", "mulch", "pebble", "speedway" ,"grass", "concrete", "brick"])
    # ar = list()
    # for i in range(cats):
    #     ar.append(labels)

    # print(cartesian_product(*ar))

































    # clusters , elbow = cluster_jackal.cluster(self.visual_encoding[idx[:2000],:])
	# 		metadata = list(zip(self.label[idx[:2000]], clusters))


	# 		print(len(self.label[idx[:2000]]))
	# 		print(len(clusters))
	# 		metadata_header = ["labels","clusters"]
    #       cluster_jackal.accuracy(self.visual_encoding[idx[:2000],:],self.label[idx[:2000]])

	# 		self.logger.experiment.add_embedding(mat=self.visual_encoding[idx[:2000], :],
	# 											 label_img=self.visual_patch[idx[:2000], :, :, :],
	# 											 global_step=self.current_epoch,
	# 											 metadata=metadata,
	# 											 metadata_header = metadata_header
	# 											)