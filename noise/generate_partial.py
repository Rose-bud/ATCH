import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os
import numpy as np
import h5py
from tqdm import tqdm
import pdb
import random
 
def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    TRAIN_LABS = []
    for item in train_labels:
        index = np.nonzero(item)[0]
        TRAIN_LABS.append(index)
    train_labels = torch.tensor(TRAIN_LABS)
    print(torch.min(train_labels))
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate
    # print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    print(partialY)
    print("Finish Generating Candidate Label Sets!\n")
    return partialY


def add_partial_to_labels(labels, noise_rate):
    num_samples, num_labels = labels.shape
    partial_labels = generate_uniform_cv_candidate_labels(labels, noise_rate)
    return partial_labels


def generate_noise_WIKI(noise):
    noise_rate = noise
    data = h5py.File('./data/tool/WIKI_partial.h5', 'r')
    for i in noise_rate:
         
        labels_matrix = np.array(list(data['LabTrain']))
        labels_matrix2 = np.array(list(data['LabTrain']))
        noisy_labels_matrix = add_partial_to_labels(labels_matrix, i)

        output_file = h5py.File('./data/partial/WIKI-lall-partial_{}.h5'.format(i), 'w')

        output_file.create_dataset('result', data=noisy_labels_matrix)
        output_file.create_dataset('True', data=labels_matrix2)

        output_file.close()

def generate_noise_nus(noise):
    noise_rate = noise
    data = h5py.File('./data/tool/NUS_partial.h5', 'r')
    for i in noise_rate:
        labels_matrix = np.array(list(data['LabTrain']))
        labels_matrix2 = np.array(list(data['LabTrain']))
        noisy_labels_matrix = add_partial_to_labels(labels_matrix, i)

        output_file = h5py.File('./data/partial/NUS-lall-partial_{}.h5'.format(i), 'w')

        output_file.create_dataset('result', data=noisy_labels_matrix)
        output_file.create_dataset('True', data=labels_matrix2)

        output_file.close()

def generate_noise_inria(noise):
    noise_rate = noise
    data = h5py.File('./data/tool/INRIA_partial.h5', 'r')
    for i in noise_rate:
        labels_matrix = np.array(list(data['LabTrain']))
        labels_matrix2 = np.array(list(data['LabTrain']))
        noisy_labels_matrix = add_partial_to_labels(labels_matrix, i)

        output_file = h5py.File('./data/partial/INRIA-lall-partial_{}.h5'.format(i), 'w')

        output_file.create_dataset('result', data=noisy_labels_matrix)
        output_file.create_dataset('True', data=labels_matrix2)

        output_file.close()

def generate_noise_xmedia(noise):
    noise_rate = noise
    data = h5py.File('./data/tool/XMEDIA_partial.h5', 'r')
    for i in noise_rate:
         
        labels_matrix = np.array(list(data['LabTrain']))
        labels_matrix2 = np.array(list(data['LabTrain']))
        noisy_labels_matrix = add_partial_to_labels(labels_matrix, i)

        output_file = h5py.File('./data/partial/XMEDIA-lall-partial_{}.h5'.format(i), 'w')
         
        output_file.create_dataset('result', data=noisy_labels_matrix)
        output_file.create_dataset('True', data=labels_matrix2)

        output_file.close()
if __name__ == "__main__":
    partial_rate_wiki = [0.1, 0.3, 0.5]
    partial_rate_nus = [0.1, 0.3, 0.5]
    partial_rate_inria = [0.01, 0.03, 0.05]
    partial_rate_xmedia = [0.1, 0.3, 0.5]

    generate_noise_WIKI(partial_rate_wiki)
    generate_noise_nus(partial_rate_nus)
    generate_noise_inria(partial_rate_inria)
    generate_noise_xmedia(partial_rate_xmedia)
