# workspace/Transformer/dataset.py

import numpy as np
import os
import tensorflow as tf


class Dataset:
    registry: list['Dataset'] = []

    def __init__(self, name, max_length_input=5_000, max_length_output=5_000):
        self.name = name
        self.base_dir = os.path.join('Training_Data', self.name)
        self.raw_path = os.path.join(self.base_dir, 'raw_data.npz')
        self.tokenized_path = os.path.join(self.base_dir, 'tokenized_data.npz')
        self.tokenized_padded_path = os.path.join(self.base_dir, 'tokenized_padded_data.npz')

        self.max_length_input = max_length_input
        self.max_length_output = max_length_output

        Dataset.registry.append(self)

    def create_dataset(self, batch_size):
        # Load
        data = np.load(self.tokenized_padded_path)

        encoder_inputs = data['encoder_inputs']
        decoder_inputs = data['decoder_inputs']
        targets = data['targets']
        # print(np.shape(problems))
        # print(np.shape(decoder_inputs))
        # print(np.shape(targets))

        # Filter to examples where both inputs and outputs are below max length
        short_problem_indices = np.where(encoder_inputs[:, self.max_length_input - 1] == 0)[0]
        short_target_indices = np.where(decoder_inputs[:, self.max_length_output - 1] == 0)[0]
        indices = np.intersect1d(short_problem_indices, short_target_indices)

        encoder_inputs = encoder_inputs[indices]
        decoder_inputs = decoder_inputs[indices]
        targets = targets[indices]

        # Create the dataset
        dataset = tf.data.Dataset.from_tensor_slices(((encoder_inputs, decoder_inputs), targets))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

Advent_of_Code = Dataset('Advent_of_Code')
Codeforces_A = Dataset('Codeforces_A')
Evol_Instruct = Dataset('Evol_Instruct')
LeetCode_Complete = Dataset('LeetCode_Complete')
LeetCode_Master = Dataset('LeetCode_Master')
LeetCode_Train = Dataset('LeetCode_Train')
Problem_Solution = Dataset('Problem_Solution')
Python_Codes = Dataset('Python_Codes')
Python_Text_to_Code = Dataset('Python_Text_to_Code')

All = Dataset('All', max_length_input=500, max_length_output=500)