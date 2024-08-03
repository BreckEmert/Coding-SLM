# Transformer/tokenizer.py

import os
import pickle

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

from Transformer import ModelArgs


class TextTokenizer:
    """Handles tokenization and padding."""

    def __init__(self, base_dir: str):
        # For saving in pickles
        self.base_dir = base_dir
        self.tokenizer_path = os.path.join(base_dir, 'tokenizer.pkl')

        # For saving metadata for embedding projection
        metadata_dir = os.path.join(base_dir, 'metadata')
        self.metadata_path = os.path.join(metadata_dir, 'metadata.tsv')
        os.makedirs(metadata_dir, exist_ok=True)

        # Instantiate keras tokenizer
        self.tokenizer = Tokenizer(filters='', oov_token='UNK', lower=True)

        # Special tokens
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.indent_token = '<INDENT>'
        self.dedent_token = '<DEDENT>'

    def save_metadata(self, tokenizer: Tokenizer, metadata_path: str) -> None:
        """Save tokenizer metadata, especially for use in embedding projection."""
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for word, index in tokenizer.word_index.items():
                f.write(f'{word}\n')

    def tokenize(
            self, problems: list[str], solutions: list[str]
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        """Tokenize problems and solutions together."""

        # Fit Keras tokenizer
        special_tokens = [self.sos_token, self.eos_token, self.indent_token, self.dedent_token]
        self.tokenizer.fit_on_texts(special_tokens + problems + solutions)

        # Save tokenizer
        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        self.save_metadata(self.tokenizer, self.metadata_path)

        # Tokenize with Keras tokenizer
        print("Length before tokenizing:", len(problems), len(solutions))
        encoder_inputs = self.tokenizer.texts_to_sequences(
            [self.sos_token + ' ' + text for text in problems])
        decoder_inputs = self.tokenizer.texts_to_sequences(
            [self.sos_token + ' ' + text for text in solutions])
        targets = self.tokenizer.texts_to_sequences(
            [text + ' ' + self.eos_token for text in solutions])

        print("Length after tokenizing:", len(encoder_inputs), len(decoder_inputs), len(targets))
        return encoder_inputs, decoder_inputs, targets

    def pad(
        self, encoder_inputs: list[list[int]], 
        decoder_inputs: list[list[int]], targets: list[list[int]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pad tokenized sequences to the same length."""
        max_length_input = max(len(seq) for seq in encoder_inputs)
        max_length_output = max(len(seq) for seq in targets)

        encoder_inputs = pad_sequences(encoder_inputs, padding='post', maxlen=max_length_input)
        decoder_inputs = pad_sequences(decoder_inputs, padding='post', maxlen=max_length_output)
        targets = pad_sequences(targets, padding='post', maxlen=max_length_output)

        print("Length after padding:", len(encoder_inputs), len(decoder_inputs), len(targets))
        return encoder_inputs, decoder_inputs, targets

    def load_tokenizer(
        self, problem_tokenizer_path: str, solution_tokenizer_path: str, args: ModelArgs
    ) -> tuple[Tokenizer, Tokenizer]:
        """Load tokenizers and update args vocabulary sizes."""
        with open(problem_tokenizer_path, 'rb') as f:
            problem_tokenizer = pickle.load(f)
            args.problem_vocab_size = len(problem_tokenizer.word_index) + 1

        with open(solution_tokenizer_path, 'rb') as f:
            solution_tokenizer = pickle.load(f)
            args.solution_vocab_size = len(solution_tokenizer.word_index) + 1

        return problem_tokenizer, solution_tokenizer