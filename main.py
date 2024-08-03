# main.py

from config_manager import ConfigManager
from Training_Data import DatasetManager
from Transformer import (
    ModelArgs, ModelManager, TextTokenizer, 
    FullWeightHistCallback, TokenProbCallback, ModelSaveCallback, TensorboardCallback
)


def main():
    """Main function to set up, train, and save the model"""
    config = ConfigManager()
    config.set_gpu_config()
    # config.tf_build_printout()
    
    # Generate the dataset and load tokenizers
    args = ModelArgs()
    dataset_manager = DatasetManager(config.base_dir, 'Python_Text_to_Code', args)  # Choose dataset here
    exit()

    tokenizer = TextTokenizer(dataset_manager.generator.base_dir)
    problem_tokenizer, solution_tokenizer = tokenizer.load_tokenizer(dataset_manager.generator, args)

    # Build the model
    model = ModelManager.build_model(config.model_path, config.checkpoint_path, args, from_checkpoint=False)
    # ModelManager.print_model_layers(model)
    # model.summary()

    # Callbacks
    weight_callback = FullWeightHistCallback(save_freq=3, log_dir=config.fit_log_dir, include_biases=True)
    token_prob_callback = TokenProbCallback(5, config.fit_log_dir, problem_tokenizer, solution_tokenizer)
    checkpoint_callback = ModelSaveCallback(save_freq=10, save_path=config.checkpoint_path)
    tensorboard_callback = TensorboardCallback(config.fit_log_dir, config.projector)
    
    # Train the model
    history = model.fit(
        dataset_manager.dataset, 
        epochs=args.epochs, 
        callbacks=[checkpoint_callback, weight_callback, token_prob_callback, tensorboard_callback]
    )
    
    # Save the model
    model.save(config.model_path)

if __name__ == '__main__':
    main()