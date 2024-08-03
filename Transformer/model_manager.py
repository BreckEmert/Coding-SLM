# Transformer/model_manager.py

import tensorflow as tf

from .model import Transformer, EncoderBlock, DecoderBlock, EncoderLayer, DecoderLayer, NoamSchedule

class ModelManager:
    """Class to manage the building and loading of the model"""

    @staticmethod
    def build_model(model_path, checkpoint_path, args, from_checkpoint):
        """Build or load the model based on the from_checkpoint flag"""
        if from_checkpoint:
            model = tf.keras.models.load_model(
                checkpoint_path,
                custom_objects={
                    'Transformer': Transformer,
                    'EncoderBlock': EncoderBlock,
                    'DecoderBlock': DecoderBlock,
                    'EncoderLayer': EncoderLayer,
                    'DecoderLayer': DecoderLayer,
                    'NoamSchedule': NoamSchedule
                }
            )
            print(f"Loaded model from checkpoint {checkpoint_path}")
        else:
            encoder_input = tf.keras.Input(shape=(None,), dtype='int32', name='encoder_input')
            decoder_input = tf.keras.Input(shape=(None,), dtype='int32', name='decoder_input')

            model = Transformer(args)
            model((encoder_input, decoder_input))
            print(f"Loaded new model to save at {model_path}")

        model.compile()
        return model
    
    def print_model_layers(self, object):
        """Recursively print the model structure"""
        if hasattr(object, 'layers'):
            print(f"{object.name} has sublayers {[layer.name for layer in object.layers]}")
            for layer in object.layers:
                self.print_model_layers(layer)