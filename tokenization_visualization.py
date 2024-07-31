import os
import pickle
import numpy as np
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from rich.console import Console
from rich.syntax import Syntax

# Set up the console
console = Console()

def load_npz_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data['encoder_inputs'], data['decoder_inputs'], data['targets']

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def detokenize(tokenizer, sequences):
    return tokenizer.sequences_to_texts(sequences)

def display_code_snippet(code_snippet):
    """Display code snippet with syntax highlighting"""
    syntax = Syntax(code_snippet, "python", theme="monokai", line_numbers=True)
    console.print(syntax)

def debug_tokenizer(tokenizer, sequences):
    word_index = tokenizer.word_index
    print(f"Word index size: {len(word_index)}")
    if "UNK" in word_index:
        print(f"UNK token index: {word_index['UNK']}")
    
    print("Example tokens and their corresponding words:")
    for sequence in sequences[:1]:  # Just checking the first sequence for debug
        for token in sequence:
            for word, index in word_index.items():
                # print(word, index)
                if index == token:
                    print(f"Token: {token}, Word: {word}")
                    break
            else:
                print(f"Token: {token}, Word: UNK")

def main():
    # Set paths
    base_dir = os.getenv('WORKSPACE_DIR', os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(base_dir, 'Training_Data', 'All')
    tokenized_data_path = os.path.join(base_dir, 'tokenized_padded_data.npz')
    problem_tokenizer_path = os.path.join(base_dir, 'problem_tokenizer.pkl')
    solution_tokenizer_path = os.path.join(base_dir, 'solution_tokenizer.pkl')

    # Load raw data
    encoder_inputs, decoder_inputs, targets = load_npz_data(tokenized_data_path)

    # Load tokenizers
    problem_tokenizer = load_tokenizer(problem_tokenizer_path)
    solution_tokenizer = load_tokenizer(solution_tokenizer_path)

    # Debug tokenizers
    print("Problem Tokenizer Debug:")
    debug_tokenizer(problem_tokenizer, encoder_inputs)

    print("\nSolution Tokenizer Debug:")
    debug_tokenizer(solution_tokenizer, decoder_inputs)

    # Detokenize data
    detokenized_problems = detokenize(problem_tokenizer, encoder_inputs[:5])
    detokenized_solutions = detokenize(solution_tokenizer, decoder_inputs[:5])

    # Visualize the data
    for i, (problem, solution) in enumerate(zip(detokenized_problems, detokenized_solutions)):
        print(f"\nExample {i+1} - Problem:\n{'-'*50}")
        display_code_snippet(problem)
        print(f"\nExample {i+1} - Solution:\n{'-'*50}")
        display_code_snippet(solution)

if __name__ == "__main__":
    main()
