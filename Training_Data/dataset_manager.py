# Training_Data/dataset_manager.py

from __future__ import annotations

import numpy as np
import tensorflow as tf
from .dataset_generator import DatasetGenerator
from .dataset import (
    Advent_of_Code, Codeforces_A, Evol_Instruct, 
    LeetCode_Complete, LeetCode_Master, LeetCode_Train, 
    Problem_Solution, Python_Codes, Python_Text_to_Code, All
)
from Transformer import ModelArgs, tokenizer


class DatasetManager:
    """Wrapper class for managing datasets and dataset generation."""

    DATASET_REGISTRY = {
        'Advent_of_Code': Advent_of_Code, 
        'Codeforces_A': Codeforces_A, 
        'Evol_Instruct': Evol_Instruct, 
        'LeetCode_Complete': LeetCode_Complete, 
        'LeetCode_Master': LeetCode_Master, 
        'LeetCode_Train': LeetCode_Train, 
        'Problem_Solution': Problem_Solution, 
        'Python_Codes': Python_Codes, 
        'Python_Text_to_Code': Python_Text_to_Code, 
        'All': All
    }

    def __init__(self, base_dir: str, dataset_name: str, args: ModelArgs):
        self.dataset_choice = self.DATASET_REGISTRY[dataset_name]
        self.tokenizer = tokenizer.TextTokenizer(base_dir)
        self.generator = DatasetGenerator(base_dir, self.tokenizer)
        self.args = args
        self.dataset = self.generate_dataset()

    def generate_dataset(self) -> None:
        """Generate the chosen dataset with args.batch_size."""
        generate_function = self.generator.get_generate_function(self.dataset_choice)
        generate_function()
        return self.create_tf_dataset()

    def create_tf_dataset(self, batch_size: int) -> tf.data.Dataset:
        """Create a tf dataset object for use in training."""
        # Load
        data = np.load(self.tokenized_padded_path)

        encoder_inputs = data['encoder_inputs']
        decoder_inputs = data['decoder_inputs']
        targets = data['targets']

        # Filter to examples where both inputs and outputs are below max length
        max_length_input = min(encoder_inputs.shape[1], self.max_length_input)
        max_length_output = min(decoder_inputs.shape[1], self.max_length_output)

        short_problem_indices = np.where(encoder_inputs[:, max_length_input - 1] == 0)[0]
        short_target_indices = np.where(decoder_inputs[:, max_length_output - 1] == 0)[0]
        indices = np.intersect1d(short_problem_indices, short_target_indices)

        encoder_inputs = encoder_inputs[indices]
        decoder_inputs = decoder_inputs[indices]
        targets = targets[indices]

        # Create the dataset
        dataset = tf.data.Dataset.from_tensor_slices(((encoder_inputs, decoder_inputs), targets))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        print("Dataset element_spec:", dataset.element_spec)

        return dataset
