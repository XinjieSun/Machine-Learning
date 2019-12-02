import numpy as np
assignment = np.array([1,2,0,0,2,0,1,0,0,2,1,2,0,0,2,2])
labels = np.array([1,2,4,3,5,6,7,9,4,5,4,1,2,3,6,5])
n_cluster = 3
vote = {n:[] for n in range(n_cluster)}
cluster_labels = []
for i in range(n_cluster):
    for j in range(len(assignment)):
        if assignment[j] == i:
            vote[i].append(labels[j])
for i in range(n_cluster):
    cluster_labels.append(np.argmax(np.bincount((np.asarray(vote[i])))))



