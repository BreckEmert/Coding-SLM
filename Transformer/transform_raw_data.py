# workspace/Transformer/transform_raw_data.py

import glob
import json
import os
import numpy as np
import pandas as pd
import random
import re
import unicodedata

from .dataset import (
    Dataset, 
    Advent_of_Code, Codeforces_A, Evol_Instruct, 
    LeetCode_Complete, LeetCode_Master, LeetCode_Train, 
    Problem_Solution, Python_Codes, Python_Text_to_Code, All
)
from .tokenizer import Tokenizers


class Dataset_Generator:
    def __init__(self, base_dir: str, choice_dir: str):
        self.base_dir = base_dir

        self.tokenizer = Tokenizers(choice_dir)
    
    def get_generate_function(self, dataset):
        return getattr(self, 'generate_' + dataset.name)

    def process_code(self, solution):
        """Remove trailing spaces and check for unicode issues"""
        # Normalize Unicode characters
        solution = unicodedata.normalize('NFKD', solution)
        
        # Remove non-ASCII characters
        solution = solution.encode('ascii', 'ignore').decode('ascii')

        processed_lines = []
        for line in solution.split("\n"):
            processed_lines.append(line)

        # Filter out leading and trailing blank lines
        try:
            while processed_lines[0].strip() == "":
                processed_lines.pop(0)
            while processed_lines[-1].strip() == "":
                processed_lines.pop()
        except IndexError:
            return None

        # Concatenate processed lines
        processed_solution = "\n".join(processed_lines)

        return processed_solution

    def generate_Advent_of_Code(self):
        # Load data
        data_path = os.path.join(self.base_dir, Advent_of_Code.base_dir, 'train_truncated.json')
        with open(data_path, 'r', encoding='utf-8') as raw_file:
            dataset = json.load(raw_file)

        # Filter for Python solutions
        python_solutions = [example for example in dataset if example['solution_lang'] == 'python']

        # Prepare the problems and solutions
        problems = []
        solutions = []
        for example in python_solutions:
            problem = (
                f"Task: {example['task']}\n\n"
                f"An example part of the input might look like {example['input'][:50]} and the answer would be {example['answer']}."
            )
            solution = example['solution']
            solution = self.process_code(solution)
            problems.append(problem)
            solutions.append(solution)

        # Tokenize and pad
        encoder_inputs = self.tokenizer.tokenize_input(problems)
        decoder_inputs, targets, _ = self.tokenizer.tokenize_output(solutions)

        # Write to npz
        self.write_file(problems, solutions, solutions, Advent_of_Code.raw_path)
        self.write_file(encoder_inputs, decoder_inputs, targets, Advent_of_Code.tokenized_path)

    def generate_Codeforces_A(self):
        # Load problems
        problems_path = os.path.join(self.base_dir, Codeforces_A.base_dir, 'A_problems.json')
        with open(problems_path, 'r') as problems_file:
            problems_list = json.load(problems_file)

        raw_problems = {}
        for problem in problems_list:
            problem_id = problem['problem_id']
            concatenated_problem = 'Problem statement: {}\n\n Example input: {}\n Example output: {}\n\n Problem notes: {}\n\n Examples: {}'.format(
                problem.get('problem_statement', ''),
                problem.get('problem_input', ''),
                problem.get('problem_output', ''),
                problem.get('problem_notes', ''),
                problem.get('examples', '')
            )
            raw_problems[problem_id] = concatenated_problem

        # Load solutions
        submissions_dir = os.path.join(self.base_dir, Codeforces_A.base_dir, 'A_submissions')
        raw_solutions = [[] for _ in range(1873)]
        submissions = glob.glob(os.path.join(submissions_dir, '*.py'))

        for submission_path in submissions:
            problem_number = int(re.findall(r'^\d+', os.path.basename(submission_path))[0])
            with open(submission_path, 'r') as submission:
                solution = submission.read()
                solution = self.process_code(solution)
                raw_solutions[problem_number].append(solution)

        # Combine problems and solutions
        problems = []
        solutions = []
        for problem_id, solution_set in enumerate(raw_solutions):
            if solution_set:
                for solution in solution_set:
                    problems.append(raw_problems[problem_id])
                    solutions.append(solution)

        # Tokenize and pad
        encoder_inputs = self.tokenizer.tokenize_input(problems)
        decoder_inputs, targets, _ = self.tokenizer.tokenize_output(solutions)

        # Write to npz
        self.write_file(problems, solutions, solutions, Codeforces_A.raw_path)
        self.write_file(encoder_inputs, decoder_inputs, targets, Codeforces_A.tokenized_path)

    def generate_Evol_Instruct(self):
        # Load data
        data_path = os.path.join(self.base_dir, Evol_Instruct.base_dir, 'Evol-Instruction-66k.json')
        with open(data_path, 'r', encoding='utf-8') as raw_file:
            dataset = json.load(raw_file)

        # Prepare the problems and solutions
        problems = []
        solutions = []
        for example in dataset:
            problem = example['instruction']
            non_ascii_count = sum(1 for char in problem if ord(char) > 127)
            if non_ascii_count > 10:
                continue

            # Find all Python code blocks and concatenate them
            python_code_matches = re.findall(r'```python\s*(.*?)\s*```', example['output'], re.DOTALL)
            if python_code_matches:
                python_code = "\n\n".join(python_code_matches)
                python_code = self.process_code(python_code)

                problems.append(problem)
                solutions.append(python_code)

        # Tokenize and pad
        encoder_inputs = self.tokenizer.tokenize_input(problems)
        decoder_inputs, targets, compiled_indices = self.tokenizer.tokenize_output(solutions)

        # Filter to only compiled solutions
        problems = [problems[i] for i in compiled_indices]
        solutions = [solutions[i] for i in compiled_indices]
        encoder_inputs = [encoder_inputs[i] for i in compiled_indices]

        # Write to npz
        self.write_file(problems, solutions, solutions, Evol_Instruct.raw_path)
        self.write_file(encoder_inputs, decoder_inputs, targets, Evol_Instruct.tokenized_path)

    def generate_LeetCode_Complete(self):
        # Load problems
        problems_path = os.path.join(self.base_dir, LeetCode_Complete.base_dir, 'leetcodecomplete.jsonl')
        with open(problems_path, 'r') as problems_file:
            dataset_list = [json.loads(line) for line in problems_file]

        problems = []
        solutions = []
        for index, example in enumerate(dataset_list):
            problems.append(example['input'])
            # Remove ```python```
            solution = re.sub(r'^```python\s*', '', example['output'].strip())
            solution = re.sub(r'\s*```$', '', solution)
            solutions.append(solution)

        # Tokenize and pad
        encoder_inputs = self.tokenizer.tokenize_input(problems)
        decoder_inputs, targets, _ = self.tokenizer.tokenize_output(solutions)

        # Write to npz
        self.write_file(problems, solutions, solutions, LeetCode_Complete.raw_path)
        self.write_file(encoder_inputs, decoder_inputs, targets, LeetCode_Complete.tokenized_path)

    def generate_LeetCode_Master(self):
        # Load problems
        problems_dir = os.path.join(self.base_dir, LeetCode_Master.base_dir, 'python_files')
        problem_files = glob.glob(os.path.join(problems_dir, '*.py'))

        problems = []
        solutions = []
        for problem_file in problem_files:
            with open(problem_file, 'r', encoding='utf-8') as file:
                content = file.read()
                # Extract question and answer parts
                match = re.search(r'"""(.*?)"""(.*)', content, re.DOTALL)
                if match:
                    question = match.group(1).strip()
                    solution = match.group(2).strip()
                    # Remove source, author, and date lines
                    question = re.sub(r'(Source : .*|Author : .*|Date   : .*)', '', question)
                    problems.append(question)
                    solutions.append(solution)
        self.write_file(problems, solutions, solutions, LeetCode_Master.raw_path)

        # Tokenize and pad
        encoder_inputs = self.tokenizer.tokenize_input(problems)
        decoder_inputs, targets, _ = self.tokenizer.tokenize_output(solutions)

        # Write to npz
        self.write_file(problems, solutions, solutions, LeetCode_Master.raw_path)
        self.write_file(encoder_inputs, decoder_inputs, targets, LeetCode_Master.tokenized_path)

    def generate_LeetCode_Train(self):
        # Load problems
        problems_path = os.path.join(self.base_dir, LeetCode_Train.base_dir, 'leetcode-train.jsonl')
        with open(problems_path, 'r') as problems_file:
            dataset_list = [json.loads(line) for line in problems_file]

        problems = []
        solutions = []

        for index, example in enumerate(dataset_list):
            problem = f"XXTITLE {example['title']} XXCONTENT {example['content']}"

            # Extract the Python code block
            python_code_match = re.search(r'```python\s*(.*?)\s*```', example['python'], re.DOTALL)
            if python_code_match:
                solution = python_code_match.group(1)
            else:
                print(f"Warning: No Python code block found in example {example['id']}")
                solution = example['python']

            problems.append(problem)
            solutions.append(solution)

        # Tokenize and pad
        encoder_inputs = self.tokenizer.tokenize_input(problems)
        decoder_inputs, targets, _ = self.tokenizer.tokenize_output(solutions)

        # Write to npz
        self.write_file(problems, solutions, solutions, LeetCode_Train.raw_path)
        self.write_file(encoder_inputs, decoder_inputs, targets, LeetCode_Train.tokenized_path)

    def generate_Problem_Solution(self):
        problems_path = os.path.join(self.base_dir, Problem_Solution.base_dir, 'Problem_Solution.csv')
        df = pd.read_csv(problems_path, encoding_errors='ignore')

        problems = []
        solutions = []
        for index, row in df.iterrows():
            problem = row['Problem']
            solution = row['Python Code']
            problems.append(problem)
            solutions.append(solution)

        # Tokenize and pad
        encoder_inputs = self.tokenizer.tokenize_input(problems)
        decoder_inputs, targets, _ = self.tokenizer.tokenize_output(solutions)

        # Write to npz
        self.write_file(problems, solutions, solutions, Problem_Solution.raw_path)
        self.write_file(encoder_inputs, decoder_inputs, targets, Problem_Solution.tokenized_path)
    
    def generate_Python_Codes(self):
        # Load data
        data_path = os.path.join(self.base_dir, Python_Codes.base_dir, 'python-codes-25k.json')
        with open(data_path, 'r', encoding='utf-8') as raw_file:
            dataset = json.load(raw_file)

        # Prepare the problems and solutions
        problems = []
        solutions = []
        for example in dataset:
            python_code_match = re.search(r'```python\s*(.*?)\s*```', example['output'], re.DOTALL)
            if python_code_match:
                python_code = python_code_match.group(1)
                python_code = self.process_code(python_code)

                if python_code is not None:
                    problem = example['instruction']
                    
                    problems.append(problem)
                    solutions.append(python_code)

        # Tokenize and pad
        encoder_inputs = self.tokenizer.tokenize_input(problems)
        decoder_inputs, targets, compiled_indices = self.tokenizer.tokenize_output(solutions)

        # Filter to only compiled solutions
        problems = [problems[i] for i in compiled_indices]
        solutions = [solutions[i] for i in compiled_indices]
        encoder_inputs = [encoder_inputs[i] for i in compiled_indices]

        # Write to npz
        self.write_file(problems, solutions, solutions, Python_Codes.raw_path)
        self.write_file(encoder_inputs, decoder_inputs, targets, Python_Codes.tokenized_path)

    def generate_Python_Text_to_Code(self):
        # Load data
        data_path = os.path.join(self.base_dir, Python_Text_to_Code.base_dir, 'combined.json')
        with open(data_path, 'r', encoding='utf-8') as raw_file:
            dataset = json.load(raw_file)

        # Prepare the problems and solutions
        problems = []
        solutions = []
        for example in dataset:
            problem = example['text']
            solution = example['code']
            solution = self.process_code(solution)
            
            problems.append(problem)
            solutions.append(solution)

        # Tokenize and pad
        encoder_inputs = self.tokenizer.tokenize_input(problems)
        decoder_inputs, targets, compiled_indices = self.tokenizer.tokenize_output(solutions)

        # Filter to only compiled solutions
        problems = [problems[i] for i in compiled_indices]
        solutions = [solutions[i] for i in compiled_indices]
        encoder_inputs = [encoder_inputs[i] for i in compiled_indices]

        # Write to npz
        self.write_file(problems, solutions, solutions, Python_Text_to_Code.raw_path)
        self.write_file(encoder_inputs, decoder_inputs, targets, Python_Text_to_Code.tokenized_path)

    def generate_All(self, force_generate=True):
        # Generate datasets if they don't exist or if force_generate
        for dataset in Dataset.registry:
            dataset_exists = os.path.exists(dataset.raw_path) and os.path.exists(dataset.tokenized_path)
            should_generate = dataset.name != "All" and (not dataset_exists or force_generate)
            
            if should_generate:
                generate_function = self.get_generate_function(dataset)
                generate_function()
        
        # Load datasets
        problems = []
        solutions = []
        all_datasets = [
            Advent_of_Code, Codeforces_A, Evol_Instruct, 
            LeetCode_Complete, LeetCode_Master, LeetCode_Train, 
            Problem_Solution, Python_Codes, Python_Text_to_Code
        ]
        for dataset in all_datasets:
            if os.path.exists(dataset.raw_path):
                data = np.load(dataset.raw_path, allow_pickle=True)
                problems.append(data['encoder_inputs'])
                solutions.append(data['decoder_inputs'])
            else:
                raise FileNotFoundError(f"{dataset.raw_path} doesn't exist for {dataset.name}")

        # Concatenate
        problems = np.concatenate(problems, axis=0)
        solutions = np.concatenate(solutions, axis=0)

        # Tokenize and pad
        encoder_inputs = self.tokenizer.tokenize_input(problems)
        decoder_inputs, targets, _ = self.tokenizer.tokenize_output(solutions)

        # Write tokenized and padded data to npz
        self.write_file(problems, solutions, solutions, All.raw_path)
        self.write_file(encoder_inputs, decoder_inputs, targets, All.tokenized_padded_path)
    
    def write_file(self, encoder_inputs, decoder_inputs, targets, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save as .npz
        np.savez_compressed(output_path, encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs, targets=targets)

        # Also save a sample as .csv
        sample_size = min(100, len(encoder_inputs))
        sample_indices = random.sample(range(len(encoder_inputs)), sample_size)
        
        encoder_sample = [encoder_inputs[i].tolist() if isinstance(encoder_inputs[i], np.ndarray) else encoder_inputs[i] for i in sample_indices]
        decoder_sample = [decoder_inputs[i].tolist() if isinstance(decoder_inputs[i], np.ndarray) else decoder_inputs[i] for i in sample_indices]
        target_sample = [targets[i].tolist() if isinstance(targets[i], np.ndarray) else targets[i] for i in sample_indices]

        data = {
            'encoder_inputs': encoder_sample,
            'decoder_inputs': decoder_sample,
            'targets': target_sample
        }

        df = pd.DataFrame(data)
        df.to_csv(output_path.replace('.npz', '.csv'), index=False)