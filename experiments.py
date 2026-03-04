# experiment.py
#
# A file to run experiments that fit in with our standard way of
# running machine learning/computer vision in Keras/TensorFow.
#
# Simon Parsons
# University of Lincoln
# 26-02-25
 
# Based heavily on suggestions from CoPilot (albeit that CoPilot wrote
# code that would not work for Sequential models).

import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
import os
from datetime import datetime

def runExperiments(
        network,
        X_train, y_train,
        X_test, y_test,
        batch_size,
        epochs,
        patience,
        validation_split,
        runs=5,
        out_file="training_results.csv",
        seed=None
    ):
    """
    arch: type of network to use.
    X_train, y_train: training arrays
    X_test, y_test: test arrays (held out)
    batch_size: batch size for .fit(), passed as a string.
    epochs: number of epochs, passed as a string (either None or a string representing an int)
    patience: how many epochs before invoking early stopping, passed as a string
    validation_split: passed to .fit()
    runs: number of repeated training runs
    out_file: CSV file to write results to
    seed: optional initial random seed for reproducibility
    """
        
    all_records = []   # list of dictionaries → becomes DataFrame
    all_summaries = [] # one for every epoch, one summarizing over a run.

    network.buildModel()
    
    for run in range(1, runs + 1):
        # Delete the existing TensorFlow environment so we can compile
        # new models without getting bogged down.
        K.clear_session()

        print(f"\n===== Starting run {run}/{runs} =====")

        # Optional reproducibility
        if seed is not None:
            tf.keras.utils.set_random_seed(seed + run)

        # Build the model, cloning if it is not the first one. This
        # allows us to safely interate with the same Sequential model.
        if run == 1:
            model = network.model
        else:
            model = tf.keras.models.clone_model(model)
        # Show the model, including the name.
        model.summary()
 
        # Compile the model
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        
        # Train. As in the parent file, we do this differently
        # depending on whether we have specified epochs or are using
        # patience.
        if epochs:
            history = model.fit(
                x=X_train,
                y=y_train,
                batch_size=batch_size,
                epochs=int(epochs),
                # The alternative is to explicitly set validation_data 
                validation_split = validation_split,
            )
        else:
            early_stopping = callbacks.EarlyStopping(patience=int(args.patience))
            history = model.fit(
                x=X_train,
                y=y_train,
                batch_size=batch_size,
                epochs=50,
                validation_split = validation_split,
                callbacks=[early_stopping]
            )

        # Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)

        # Print test metrics
        print("Test loss      :", test_metrics['loss'])
        print("Test accuracy  :", test_metrics['accuracy'])

        d = {'x':1, 'y':2, 'z':3}
        d1 = {'x':'a', 'y':'b', 'z':'c'}
        # Store per-epoch results. These are pulled from the history
        # and will differ from those reported while the model is
        # training.
        hist = history.history  # dict of lists: { "loss": [...], "accuracy": [...], ... }
        for epoch in range(int(epochs)):
            record = {"run": run, "epoch": epoch + 1}
            
            # Training + validation metrics
            for key, values in hist.items():
                record[key] = values[epoch]

            all_records.append(record)

        # Store one summary row per run
        run_summary = {"run": run}
        run_summary.update(test_metrics)
        for k, v in hist.items():
            if len(v) > 0:
                run_summary[f"final_{k}"] = v[-1]
        all_summaries.append(run_summary)

    # After runs, convert records to dataframe
    df_epochs = pd.DataFrame(all_records)
    df_summary = pd.DataFrame(all_summaries)
    # Update df column names so that test values are clear in the CSV file
    df_summary.columns = ['run', 'test accuracy', 'test loss', 'final_accuracy', 'final_loss', 'final_val_accuracy', 'final_val_loss']

    # Timestamp filenames to avoid over-writing. We put the per epoch
    # results in out_file and create an additional _summary file for
    # the summary results.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_epoch = f"{os.path.splitext(out_file)[0]}_{timestamp}.csv"
    out_file_summary = f"{os.path.splitext(out_file)[0]}_{"summary"}.csv"
    filename_summary = f"{os.path.splitext(out_file_summary)[0]}_{timestamp}.csv"

    # Save to disk
    df_epochs.to_csv(filename_epoch, index=False)
    df_summary.to_csv(filename_summary, index=False)

    print(f"\nPer-epoch results saved to: {filename_epoch}")
    print(f"Per-run summary saved to:  {filename_summary}")

    return df_epochs, df_summary
