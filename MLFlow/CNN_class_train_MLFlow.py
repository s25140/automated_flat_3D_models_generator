import mlflow
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf

import mlflow.keras

# Load dataset
print("Loading dataset...")
images = np.load("Datasets/images_class_norm.npy")
labels = np.load("Datasets/labels_class.npy")
images, labels = shuffle(images, labels, random_state=25140)
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=25140)

# Splitting into batches
from tensorflow import convert_to_tensor
from tensorflow.keras.utils import Sequence
class DataGenerator(Sequence):
    def __init__(self, images, labels, batch_size):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_images = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return convert_to_tensor(batch_images), convert_to_tensor(batch_labels)
    
    def on_epoch_end(self):
        self.images, self.labels = shuffle(self.images, self.labels)

train_images_gen = DataGenerator(X_train, y_train, 32)
test_images_gen = DataGenerator(X_test, y_test, 32) 

from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import optuna
from optuna.integration import TFKerasPruningCallback


def create_model(num_filters=128, dropout_rate=0.6, dense_units=32):
    # Input for the source image
    source_input = Input(shape=(256, 256, 1), name="source_image")

    # First Conv2D layer
    x = Conv2D(8, (5, 5), activation="relu", padding="same", strides=(1, 1), use_bias=True)(source_input)

    # Merge the source image with the output of the first layer
    merged = Concatenate(axis=-1)([x, source_input])

    # Second Conv2D layer
    x = Conv2D(num_filters, (5, 5), activation="relu", padding="same", strides=(3, 3), use_bias=True)(merged)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling2D((2, 2), strides=2, padding='valid')(x)

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(int(dense_units/2), activation="relu", use_bias=True)(x)
    x = Dense(dense_units, activation="relu", use_bias=True)(x)
    output = Dense(1, activation="sigmoid", use_bias=True)(x)

    model = Model(inputs=source_input, outputs=output)
    model.summary()
    return model

# assign custom weights to the first Conv layer
def assign_custom_weights_to_Conv2D_layer(model):
    conv_layer_with_custom_weights = model.layers[1]
    model_weights = conv_layer_with_custom_weights.get_weights()[0]
    for x in range(5):
        for y in range(5):
            model_weights[x,y,0,0] = (4-x)/4
            model_weights[y,x,0,1] = (4-x)/4
            model_weights[x,y,0,2] = (x)/4
            model_weights[y,x,0,3] = (x)/4
            # corner gradients
            color = -(4-x)*(4-y)/16 if x > 1 or y > 1 else 0
            model_weights[x,y,0,4] = color
            model_weights[4-y,4-x,0,5] = color
            model_weights[4-y,x,0,6] = color
            model_weights[y,4-x,0,7] = color
            
    model.layers[1].set_weights([model_weights, np.zeros(8)])
    return model

# Training function
def train_model(optimize_hyperparameters=False, mlflow_name=None):
    model = create_model(dropout_rate=0.49496)

    #model = assign_custom_weights_to_Conv2D_layer(model)
    
    optimizer = SGD(learning_rate=0.001438, momentum=0.767624)

    precision = Precision(name='precision')
    recall = Recall(name='recall')
    def f1_score(y_true, y_pred):
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    metrics=["accuracy", precision, recall, f1_score]
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=metrics
    )
    
    #example_input = pd.DataFrame(next(iter(train_images_gen))[0]) # problem with batch size 32

    with mlflow.start_run():
        if mlflow_name is not None:
            mlflow.set_tag("mlflow.runName", mlflow_name)
        mlflow.keras.autolog()
        
        if not optimize_hyperparameters:
            checkpoint_path = "tmp/checkpoint_two_inputs_GPU"
            model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)
            early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
            model.fit(train_images_gen, batch_size=32, epochs=15, verbose=1, validation_data=test_images_gen, callbacks=[model_checkpoint, early_stopping])

            signature = None #mlflow.models.signature.infer_signature(example_input, model.predict(example_input))

            mlflow.keras.log_model(model, "cnn_class_model", signature=signature)
        else:
            def objective(trial):
                # Define hyperparameters to tune
                learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
                momentum = trial.suggest_uniform('momentum', 0.1, 0.9)
                dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.7)
                num_filters = trial.suggest_categorical('num_filters', [8, 16, 32, 64, 128])
                batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
                dense_units = trial.suggest_categorical('dense_units', [16, 32, 64, 128])

                model = create_model(num_filters=num_filters, dropout_rate=dropout_rate, dense_units=dense_units)
                model = assign_custom_weights_to_Conv2D_layer(model)
                optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
                model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)

                model_checkpoint = ModelCheckpoint(filepath="tmp/checkpoint_optuna", save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)
                early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
                history = model.fit(train_images_gen, batch_size=batch_size, epochs=15, verbose=1, validation_data=test_images_gen, callbacks=[model_checkpoint, early_stopping, TFKerasPruningCallback(trial, 'val_accuracy')])

                val_accuracy = history.history['val_accuracy'][-1]
                return val_accuracy

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=60)

            print("Best hyperparameters: ", study.best_params)

if __name__ == "__main__":
    train_model(optimize_hyperparameters=False, mlflow_name="CNN_no_custom_weights")