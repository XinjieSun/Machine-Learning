import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here
	for line in train_data:
		line.words = [line.words[i].lower() for i in range(len(line.words))]
	nums_state = len(tags)
	state_dict = {i: j for i, j in zip(tags, range(nums_state))}
	obs_dict = {}
	td_index = 0
	for line in train_data:
		for word in line.words:
			if word not in obs_dict.keys():
				obs_dict[word] = td_index
				td_index += 1
	num_obs_symbol = len(obs_dict)
	pi = np.zeros(nums_state)
	A = np.zeros([nums_state, nums_state])
	B = np.zeros([nums_state, num_obs_symbol])
	count = dict(zip(tags, np.zeros(nums_state)))
	for tag in tags:
		for line in train_data:
			if line.tags[0] == tags[state_dict[tag]]:
				count[tag] += 1
	for tag in tags:
		pi[state_dict[tag]] = count[tag]/len(train_data)
	for line in train_data:
		L = len(line.words)
		for i in range(L-1):
			A[state_dict[line.tags[i]]][state_dict[line.tags[i+1]]] += 1
		for i in range(L):
			B[state_dict[line.tags[i]]][obs_dict[line.words[i]]] += 1
	A = A / np.sum(A, axis=1)
	B = (B.T / np.sum(B, axis=1)).T
	model = HMM(pi, A, B, obs_dict, state_dict)
	###################################################
	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	for line in test_data:
		line.words = [line.words[i].lower() for i in range(len(line.words))]
	for line in test_data:
		for word in line.words:
			if word not in model.obs_dict.keys():
				model.obs_dict[word] = len(model.obs_dict)
				model.B = np.append(model.B, np.ones([len(tags), 1])*1e-6, axis=1)
		tagging.append(model.viterbi(line.words))
	###################################################
	return tagging
