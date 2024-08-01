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

def load_npz(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data['encoder_inputs'], data['decoder_inputs'], data['targets']

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def detokenize(tokenizer, sequences):
    return tokenizer.sequences_to_texts(sequences)

def display_code(code):
    """Display a piece of code with syntax highlighting"""
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(syntax)

def debug_tokenizer(tokenizer, sequences):
    """Checks vocab, UNK token, and displays a detokenized sequence"""
    word_index = tokenizer.word_index
    print(f"Word index size: {len(word_index)}")
    if "UNK" in word_index:
        print(f"UNK token index: {word_index['UNK']}")
    
    print("Example tokens and their corresponding words:")
    for sequence in sequences[:1]:  # Just checking the first sequence for debug
        for token in sequence:
            if token == 0:
                continue

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
    encoder_inputs, decoder_inputs, targets = load_npz(tokenized_data_path)

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
        display_code(problem)
        print(f"\nExample {i+1} - Solution:\n{'-'*50}")
        display_code(solution)

if __name__ == "__main__":
    main()
