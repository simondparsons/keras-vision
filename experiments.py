# experiment.py
#
# A file to run experiments that fit in with our standard way of
# running machine learning/computer vision in Keras/TensorFow.
#
# Simon Parsons
# University of Lincoln
# 26-02-25

# Based heavily on suggestions from CoPilot
import numpy as np
import pandas as pd
import tensorflow as tf
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
    model_fn: a function that returns a NEW compiled model each run.
    X_train, y_train: training arrays
    X_test, y_test: test arrays (held out)
    batch_size: batch size for .fit(), passed as a string.
    epochs: number of epochs, passed as a string
    patience: how long to tolerate increasing validation error, passed as a string
    validation_split: passed to .fit()
    runs: number of repeated training runs
    out_file: CSV file to write results to
    seed: optional initial random seed for reproducibility
    """
    
    all_records = []   # list of dictionaries → becomes DataFrame

    for run in range(1, runs + 1):
        print(f"\n===== Starting run {run}/{runs} =====")

        # Optional reproducibility
        if seed is not None:
            tf.keras.utils.set_random_seed(seed + run)

        # Build a fresh model
        network.buildModel()
        print(network.model.name)
        model = network.model
        model.summary()
        
        # Compile the model
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        
        # Train. As in the parent file, we do this differently
        # depending on whether we have specified opchs or are using
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
        test_metrics = model.evaluate(X_test, y_test, verbose=0)
        metric_names = model.metrics_names
        test_metrics_dict = {f"test_{name}": value for name, value in zip(metric_names, test_metrics)}

        # Store per-epoch results
        hist = history.history
        for epoch in range(int(epochs)):
            record = {
                "run": run,
                "epoch": epoch + 1
            }

            # Training + validation metrics
            for key, values in hist.items():
                record[key] = values[epoch]

            # Add test metrics (same for every epoch of run)
            record.update(test_metrics_dict)

            all_records.append(record)

    # Convert to dataframe
    df = pd.DataFrame(all_records)

    # Add timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{os.path.splitext(out_file)[0]}_{timestamp}.csv"

    df.to_csv(filename, index=False)
    print(f"\nAll results saved to: {filename}")

    return df
