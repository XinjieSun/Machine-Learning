import numpy as np
from data_process import Dataset

data = Dataset("pos_tags.txt", "pos_sentences.txt", train_test_split=0.8, seed=0)

train_data = data.words
#obs_dict = {i: j for i, j in zip(data.train_data, range(len(data.train_data)))}
np_array = np.array(data.train_data)
print(data.tags)
print(np_array)