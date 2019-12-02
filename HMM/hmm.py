from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        ob_index = np.array([self.obs_dict[i] for i in Osequence])
        alpha_transfer = alpha.T
        for i in range(S):
            alpha_transfer[0][i] = self.pi[i]*self.B[i][ob_index[0]]
        for m in range(1, L):
            alpha_transfer[m] = self.B[:, ob_index[m]]*np.dot(alpha_transfer[m-1], self.A)
        alpha = alpha_transfer.T
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        ob_index = np.array([self.obs_dict[i] for i in Osequence])
        beta_transfer = beta.T
        beta_transfer[L-1] = 1
        for m in range(L-2, -1, -1):
            beta_transfer[m] = np.dot(self.A, (beta_transfer[m+1] * self.B[:, ob_index[m+1]]))
        beta = beta_transfer.T
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        transfer_alpha = alpha.T
        prob = np.sum(transfer_alpha[len(Osequence)-1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        probability = self.sequence_prob(Osequence)
        prob = alpha*beta/probability
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        probability = self.sequence_prob(Osequence)
        ob_index = np.array([self.obs_dict[i] for i in Osequence])
        for i in range(S):
            for j in range(S):
                for k in range(L-1):
                    prob[i][j][k] = alpha[i][k]*self.A[i][j]*self.B[j][ob_index[k+1]]*beta[j][k+1]/probability
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        ob_index = np.array([self.obs_dict[i] for i in Osequence])
        S = len(self.pi)
        L = len(Osequence)
        prob_matrix = np.zeros([L, S])
        prob_matrix[0] = self.pi * self.B[:, ob_index[0]]
        for i in range(1, L):
            prob_matrix[i] = self.B[:, ob_index[i]] * np.max(self.A * np.expand_dims(prob_matrix[i-1], 1), axis=0)
        path_ind = np.zeros(L)
        r_st_dict = {v: k for k, v in self.state_dict.items()}
        path_ind[L-1] = np.argmax(prob_matrix[L-1])
        for t in range(L-2, -1, -1):
            path_ind[t] = np.argmax(self.A[:, int(path_ind[t+1])]*prob_matrix[t])
        path = [r_st_dict[ind] for ind in path_ind]
        # print(r_st_dict)
        #print(self.state_dict)
        #print(self.obs_dict)
        ###################################################
        return path
