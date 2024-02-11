#!/usr/bin/env python
# coding: utf-8


import datetime
import os

import tensorflow as tf
import pickle

from Transformer import ModelArgs, Loader, Transformer, build_and_compile, load_data


def main():
    base_log_dir = os.path.join("/workspace/logs", "run_" + datetime.datetime.now().strftime("%m_%d_%H_%M"))
    fit_log_dir = os.path.join(base_log_dir, "fit")
    debug_log_dir = os.path.join(base_log_dir, "debug")

    """
    # Enable full debugging
    #tf.config.optimizer.set_jit(False)  # Disable XLA compilation
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode() # Disables tf.data eager execution; not covered by run_functions_eagerly

    tf.debugging.experimental.enable_dump_debug_info(
        debug_log_dir,
        tensor_debug_mode="NO_TENSOR", # CONCISE_HEALTH
        circular_buffer_size=1000) # 1000
    """
    
    args = ModelArgs()
    
    # Initialize the Loader
    dataset_choice = "All" # [All, CodeForces_A_difficulty, ProblemSolutionPythonV3]
    dataset_path = f"/workspace/Training_Dat/{dataset_choice}.npz"
    loader = Loader(dataset_path, args)
    loader.create_dataset()
    
    # Load tokenizer information
    with open('problem_tokenizer.pkl', 'rb') as f:
        problem_tokenizer = pickle.load(f)
        args.problem_vocab_size = len(loader.problem_tokenizer.word_index) + 1
    
    with open('solution_tokenizer.pkl', 'rb') as f:
        solution_tokenizer = pickle.load(f)
        args.solution_vocab_size = len(loader.solution_tokenizer.word_index) + 1
    
    # Build the model
    model = build_and_compile(args)
    
    # Setup TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fit_log_dir, histogram_freq=1)
    
    # Train the model
    history = model.fit(loader.dataset, epochs=args.epochs, callbacks=[tensorboard_callback]) # history variable unused...
    
    """
    Manual training setup
    
    optimizer = model.optimizer
    for epoch in range(epochs):
        print(f"Start of Epoch {epoch+1}")
        
        # Iterate over the batches of the dataset.
        for step, ((tokenized_question, tokenized_code), target) in enumerate(loader.dataset):
            # Call the custom train_step
            loss = train_step(model, optimizer, tokenized_question, tokenized_code[:, :-1])
            
            # Log every 200 batches
            if step % 200 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.numpy()}")
        
        print(f"End of Epoch {epoch+1}, Loss: {loss.numpy()}")
    """
    
    # Save the model
    model.save("/workspace")

if __name__ == "__main__":
    main()