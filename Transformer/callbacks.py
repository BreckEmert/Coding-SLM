# Transformer/callbacks.py

import os
import tensorflow as tf

class FullWeightHistCallback(tf.keras.callbacks.Callback):
    """Callback to save histograms of every trainable model weight"""

    def __init__(self, save_freq, log_dir, include_biases):
        super(FullWeightHistCallback, self).__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.save_freq = save_freq
        self.include_biases = include_biases

    def save_histograms(self, epoch):
        if (epoch+1) % self.save_freq:
            with self.file_writer.as_default():
                for layer in self.model.layers:
                    self.log_layer_histograms(layer, epoch) # Recursive
            self.file_writer.flush()
    
    def log_layer_histograms(self, layer, epoch):
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                self.log_layer_histograms(sublayer, epoch)
        elif hasattr(layer, 'weights'):
            for weight in layer.weights:
                if 'bias' not in weight.name or self.include_biases:
                    tf.summary.histogram(f"Weights/{layer.name}", weight, step=epoch)

    def on_epoch_end(self, epoch, logs=None):
        self.save_histograms(epoch)


class TokenProbCallback(tf.keras.callbacks.Callback):
    """Callback to test quality of specific outputs and health of probability distribution"""

    def __init__(self, save_freq, log_dir, problem_tokenizer, solution_tokenizer):
        super().__init__()
        self.save_freq = save_freq
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.problem_tokenizer = problem_tokenizer
        self.solution_tokenizer = solution_tokenizer

        sample_encoder_input = ["Using python, write a for loop that prints 10 numbers."]
        sample_decoder_input = ["for i in range("]
        self.sample_encoder_input = self.preprocess_string(sample_encoder_input)
        self.sample_decoder_input = self.preprocess_string(sample_decoder_input)

        self.example_word_1 = "10"
        self.example_word_2 = "<EOS>"
        self.example_index_1 = self.get_token_indices(self.example_word_1)
        self.example_index_2 = self.get_token_indices(self.example_word_2)

    def preprocess_string(self, input_string):
        tokenized_input = self.problem_tokenizer.texts_to_sequences(input_string)
        tensor_input = tf.convert_to_tensor(tokenized_input)
        return tensor_input

    def get_token_indices(self, word):
        index = self.solution_tokenizer.word_index.get(word)
        print(f"TokenProbabilityCallback example word {word}: {index}")
        return index

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%self.save_freq == 0:
            predictions = self.model.predict((self.sample_encoder_input, self.sample_decoder_input))
            token_probs = tf.nn.softmax(predictions, axis=-1)
            with self.file_writer.as_default():
                filtered_probs = tf.boolean_mask(token_probs, token_probs > 0.01) # Trim down
                tf.summary.histogram(f"token_probabilities", filtered_probs, step=epoch)
                
                token_prob = tf.reduce_mean(token_probs[:, :, self.example_index_1])
                tf.summary.scalar(f"token_probabilities/{self.example_word_1}", token_prob, step=epoch)

                token_prob = tf.reduce_mean(token_probs[:, :, self.example_index_2])
                tf.summary.scalar(f"token_probabilities/{self.example_word_2}", token_prob, step=epoch)


class ModelSaveCallback(tf.keras.callbacks.Callback):
    """Callback to save the model to a checkpoint"""

    def __init__(self, save_freq, save_path):
        super(ModelSaveCallback, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.save_freq == 0:
            self.model.save(self.save_path)
            print(f"\nSaved model checkpoint for epoch {epoch+1}")


class TensorboardCallback(tf.keras.callbacks.TensorBoard):
    """TensorBoard callback with embeddings metadata"""

    def __init__(self, log_dir, projector_dir):
        super(TensorboardCallback, self).__init__(
            log_dir=log_dir,
            histogram_freq=1,
            embeddings_freq=1,
            embeddings_metadata={
                'problem_embedding': os.path.join(projector_dir, 'problem_metadata.tsv'),
                'solution_embedding': os.path.join(projector_dir, 'solution_metadata.tsv')
            }
        )