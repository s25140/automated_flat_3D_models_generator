from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

import mlflow.keras
import optuna
from optuna.integration import TFKerasPruningCallback

# Load dataset
print("Loading dataset...")
cropped_images_features = pd.read_parquet('./huggingface_tests/cropped_images_features.parquet')
not_cropped_images_features = pd.read_parquet('./huggingface_tests/not_cropped_images_features.parquet')
# combine the two dataframes
df = pd.concat([cropped_images_features, not_cropped_images_features])
# mix the rows
df = df.sample(frac=1).reset_index(drop=True)
print("df shape:", df.shape)

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['label']), df['label'], test_size=0.2, random_state=25140)



# Define the model
def create_model(dropout_rate=0.533):
    
    model = Sequential([
        Flatten(input_shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid'),
    ])

    model.summary()
    return model


# Training function
def train_model(optimize_hyperparameters=False, mlflow_name=None):
    model = create_model(dropout_rate=0.49496)
    
    optimizer = SGD(learning_rate=0.001438, momentum=0.767624)
    #optimizer = SGD(learning_rate=0.01, momentum=0.1)
    #optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    
    #example_input = pd.DataFrame(next(iter(train_images_gen))[0]) # problem with batch size 32

    with mlflow.start_run():
        if mlflow_name is not None:
            mlflow.set_tag("mlflow.runName", mlflow_name)
        mlflow.keras.autolog()
        
        checkpoint_path = "models/tmp/checkpoint_AIMv2_NN_GPU"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        if not optimize_hyperparameters:
            model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)
            early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
            model.fit(X_train, y_train, batch_size=32, epochs=15, verbose=1, validation_data=(X_test, y_test), callbacks=[model_checkpoint, early_stopping])

            signature = None #mlflow.models.signature.infer_signature(example_input, model.predict(example_input))

            mlflow.keras.log_model(model, "AIMv2_NN_class_model", signature=signature)
        else:
            def objective(trial):
                # Define hyperparameters to tune
                learning_rate = 0.059437608892229515#trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
                momentum = 0.5938654934947398#trial.suggest_uniform('momentum', 0.1, 0.9)
                dropout_rate = 0.533#trial.suggest_uniform('dropout_rate', 0.2, 0.7)
                batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])


                model = create_model(dropout_rate=dropout_rate)
                optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
                model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

                model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)
                early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
                history = model.fit(X_test, y_test, batch_size=batch_size, epochs=20, validation_data=(X_test, y_test), callbacks=[model_checkpoint, early_stopping, TFKerasPruningCallback(trial, 'val_accuracy')])

                val_accuracy = history.history['val_accuracy'][-1]
                return val_accuracy

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=6)

            print("Best hyperparameters: ", study.best_params)

if __name__ == "__main__":
    train_model(optimize_hyperparameters=True, mlflow_name="AIMv2_NN_class_with_optimized_hyperparameters")