# config_manager.py

import datetime
import os

import tensorflow as tf

class ConfigManager:
    """Utility class for main.py"""
    
    def __init__(self):
        """Holds paths"""
        self.base_dir = os.getenv('WORKSPACE_DIR', os.path.dirname(os.path.abspath(__file__)))

        self.model_dir = os.path.join(self.base_dir, "Transformer", "model_files")
        self.model_path = os.path.join(self.model_dir, "trained_model.keras")
        self.checkpoint_path = os.path.join(self.model_dir, "checkpoint.keras")

        time = (datetime.datetime.now() - datetime.timedelta(hours=5)).strftime("%m%d_%H%M")
        self.base_log_dir = os.path.join(self.model_dir, 'logs', 'run_' + time)
        self.fit_log_dir = os.path.join(self.base_log_dir, 'fit')
    
    @staticmethod
    def set_gpu_config():
        """Ensure GPU is available and enable memory growth"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_visible_devices(gpus[0], 'GPU')
                print(f"Using GPU: {gpus[0]}")
            except RuntimeError as error:
                print(error)
        else:
            print("No GPU found")
            input("Press enter to acknowledge that no GPU is available")

    @staticmethod
    def tf_build_printout():
        print("Get build info", tf.sysconfig.get_build_info())
        print("tf version:", tf.__version__)
        print("cudnn version:", tf.sysconfig.get_build_info()["cudnn_version"])
        print("cuda version:", tf.sysconfig.get_build_info()["cuda_version"])
        print("Num GPUs Available: ", tf.config.list_physical_devices())