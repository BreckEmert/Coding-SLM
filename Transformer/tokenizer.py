# workspace/Transformer/tokenizer.py

import io
import os
import re
import pickle
import tokenize

from nltk.tokenize import word_tokenize # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


class Tokenizers:
    def __init__(self, base_dir):
        # For saving in pickles
        self.base_dir = base_dir
        self.problem_tokenizer_path = os.path.join(base_dir, 'problem_tokenizer.pkl')
        self.solution_tokenizer_path = os.path.join(base_dir, 'solution_tokenizer.pkl')

        # For saving metadata for embedding projection
        metadata_dir = os.path.join(base_dir, 'metadata')
        self.problem_metadata_path = os.path.join(metadata_dir, 'problem_metadata.tsv')
        self.solution_metadata_path = os.path.join(metadata_dir, 'solution_metadata.tsv')
        os.makedirs(metadata_dir, exist_ok=True)

        # Instantiate keras tokenizer
        self.problem_tokenizer = Tokenizer(filters='', oov_token='UNK')
        self.solution_tokenizer = Tokenizer(filters='', oov_token='UNK', lower=False)

        # Special tokens
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.indent_token = '<INDENT>'
        self.dedent_token = '<DEDENT>'
        self.hardcoded_tokens = [
            '.', ',', '(', ')', ';', '!', '"', '-'
        ]

    def save_metadata(self, tokenizer, metadata_path):
        with open(metadata_path, 'w') as f:
            for word, index in tokenizer.word_index.items():
                f.write(f'{word}\n')
    
    def add_hardcoded_tokens(self, text):
        # Ensure hardcoded tokens are split correctly
        for token in self.hardcoded_tokens:
            text = re.sub(r'(\s*{}\s*)'.format(re.escape(token)), r' \1 ', text)
        return text
    
    def tokenize_input(self, problems):
        # Fit Keras tokenizer
        self.problem_tokenizer.fit_on_texts(problems)

        # Save tokenizer
        with open(self.problem_tokenizer_path, 'wb') as f:
            pickle.dump(self.problem_tokenizer, f)
        self.save_metadata(self.problem_tokenizer, self.problem_metadata_path)

        # Tokenize with Keras tokenizer
        encoder_inputs = self.problem_tokenizer.texts_to_sequences(problems)
        
        # Pad to same length
        max_length_input = max(len(seq) for seq in encoder_inputs)
        encoder_inputs = pad_sequences(encoder_inputs, padding='post', maxlen=max_length_input)
        
        return encoder_inputs
    
    def tokenize_output(self, solutions):
        # Tokenize with Python tokenizer
        decoder_inputs = []
        targets = []
        compiled_indices = []
        for index, solution in enumerate(solutions):
            tokens = []
            current_line = None
            try:
                for token in tokenize.generate_tokens(io.StringIO(solution).readline):
                    # Skip #-based comments (multiline skipped later)
                    if token.type == tokenize.COMMENT:
                        current_line = token.start[0]
                    if current_line and token.start[0] == current_line:
                        continue # Skip
                    else:
                        current_line = None # No longer in a comment
                    
                    # Special process for spacing and tokenize strings
                    if token.type == tokenize.STRING:
                        string = token.string
                        
                        if string.startswith('"""') or string.startswith("'''"):
                            continue # Skip multi-line comments
                        
                        # Add quotes and content to tokens in order
                        tokens.append(string[0])
                        tokens.extend(word_tokenize(string[1:-1]))
                        tokens.append(string[-1])
                    elif token.type == tokenize.INDENT:
                        tokens.append(self.indent_token)
                    elif token.type == tokenize.DEDENT:
                        tokens.append(self.dedent_token)
                    else:
                        tokens.append(token.string)
                
                # Remove extra lines and spaces at the end
                while tokens and tokens[-1] in ('', self.indent_token, self.dedent_token):
                    tokens.pop()
            except (tokenize.TokenError, IndentationError) as e:
                continue # Errors counted via compiled_indices

            compiled_indices.append(index)
            decoder_inputs.append([self.sos_token] + tokens)
            targets.append(tokens + [self.eos_token])
        
        print(f"ERRORS COUNT: {len(solutions) - len(compiled_indices)}")
        
        # Fit Keras tokenizer and tokenize
        special_tokens = [self.sos_token, self.eos_token, self.indent_token]
        self.solution_tokenizer.fit_on_texts([special_tokens] + decoder_inputs + [targets[-1]])

        # Save tokenizer
        with open(self.solution_tokenizer_path, 'wb') as f:
            pickle.dump(self.solution_tokenizer, f)
        self.save_metadata(self.solution_tokenizer, self.solution_metadata_path)

        # Tokenize with Keras tokenizer
        decoder_inputs = self.solution_tokenizer.texts_to_sequences(decoder_inputs)
        targets = self.solution_tokenizer.texts_to_sequences(targets)

        # Pad
        max_length_output = max(len(seq) for seq in targets)
        decoder_inputs = pad_sequences(decoder_inputs, padding='post', maxlen=max_length_output)
        targets = pad_sequences(targets, padding='post', maxlen=max_length_output)

        return decoder_inputs, targets, compiled_indices