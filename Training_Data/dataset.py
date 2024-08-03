# Transformer/dataset.py

import os

class Dataset:
    registry: list['Dataset'] = []

    def __init__(self, name, max_length_input=5_000, max_length_output=5_000):
        self.name = name
        self.base_dir = os.path.join('Training_Data', 'Individual_Datasets', self.name)
        self.raw_path = os.path.join(self.base_dir, 'raw_data.npz')
        self.tokenized_path = os.path.join(self.base_dir, 'tokenized_data.npz')
        self.tokenized_padded_path = os.path.join(self.base_dir, 'tokenized_padded_data.npz')

        self.max_length_input = max_length_input
        self.max_length_output = max_length_output

        Dataset.registry.append(self)

Advent_of_Code = Dataset('Advent_of_Code')
Codeforces_A = Dataset('Codeforces_A')
Evol_Instruct = Dataset('Evol_Instruct')
LeetCode_Complete = Dataset('LeetCode_Complete')
LeetCode_Master = Dataset('LeetCode_Master')
LeetCode_Train = Dataset('LeetCode_Train')
Problem_Solution = Dataset('Problem_Solution')
Python_Codes = Dataset('Python_Codes')
Python_Text_to_Code = Dataset('Python_Text_to_Code')

All = Dataset('_All', max_length_input=500, max_length_output=500)