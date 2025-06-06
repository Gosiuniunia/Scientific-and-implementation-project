"""

This script:

- performs a transfer learning of CNN VGG16 model, using provided image data, so it can perform seasonal beauty type classification
- VGG16 model has only an output layer changed so it's adjusted for 4 classes classification problem
- collects model training, evaluation and classification statistics like accuracy, precision, recall and F1 score
- Kfolds validation is used in the learning process - there is k models trained, each time with different train and test data, indicated by the file fold.assignments.csv

"""
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import classification_report
import csv
from keras.utils import set_random_seed
import os

set_random_seed(42)

import tensorflow as tf
import numpy as np
import random

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)


def split_data_test_train(assignment_file_path, non_augmented_assignment_file_path, fold, offset=0):
    """
    Splits images details (file path, label), provided as pandas Dataframe, into train and test sets details pandas Dataframes,
    taking into account the fold number k
    Args:
        assignment_file_path - path to folds assignment csv file
        non_augmented_assignment_file_path - path to assignment to folds of non-augmented images stored in a csv file
        fold: number of current fold
        offset: offset for current fold. Should be set to 0 for running experiment the first time and 5 if running for the second time

    Returns:
        train_df: train data details pandas Dataframe, including label and file path
        test_df:  test data details pandas Dataframe, including label and file path

    """
    # Retrieving and adjusting folds assignments details
    df = pd.read_csv(assignment_file_path)
    non_augmented_df = pd.read_csv(non_augmented_assignment_file_path)
    df['filename'] = df['label'] + '/' + df['filename']
    non_augmented_df['filename'] = non_augmented_df['label'] + '/' + non_augmented_df['filename']

    train_df = df[df['kfold']+offset != fold]
    test_df = non_augmented_df[non_augmented_df['kfold']+offset == fold]

    return test_df, train_df

def prepare_vgg16_model():
    """

    Function sets up a VGG16 model with Imagenet pre-trained weights for 4 classes classification task.
    Model reference: https://keras.io/api/applications/vgg/

    Returns:
        model: VGG16 model

    """
    # model definition
    input_shape = (224, 224, 3)
    num_classes = 4
    base_model = VGG16(include_top=True, weights='imagenet', input_shape=input_shape)
    x = base_model.layers[-2].output
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    for layer in base_model.layers:
                layer.trainable = False

    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
    return model

def adjust_folds_assignment_file(assignment_file_input_path, assignment_file_output_path, prefixes_list):
    """
    Function modifies prepares the folds assignment for the augmented data.
    Args:
        assignment_file_input_path: path to a .csv file with folds assignment information
        assignment_file_output_path: path where adjusted .csv file with fold assignment information will be saved
        prefixes_list: list of prefixes which are used in augmented images files, for example "co" for cut-out augmented images files

    Returns:
        None
    """
    with open(assignment_file_input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        header = reader.fieldnames

        new_rows = []
        for row in rows:
            original_filename = row["filename"]
            new_rows.append(row)
            for prefix in prefixes_list:
                new_row = row.copy()
                new_row["filename"] = prefix + original_filename
                new_rows.append(new_row)

    with open(assignment_file_output_path, mode='w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(new_rows)

def run_deep_learning(images_path, model_free_images_path, folds_assignment_path, model_free_folds_assignment_path, offset_val=0, current_approach="basic_shuffle_with_seed"):

    """
    Function runs training of the VGG16 model for PCoA task.
    Args:
        images_path: path to original images
        model_free_images_path: path to images with model free image augmentaation applied
        folds_assignment_path: path to the csv file containing fold assigment information
        model_free_folds_assignment_path: path to the csv file containing fold assignment information when using model free image augmentation
        offset_val: integer value needed when experiment is run twice to change the numeration of folds.
        current_approach: current approach to train model.
        Value should be set to "basic_shuffle_with_seed" when training without data augmentation applied is intended.
        Value should be set to "model_free_shuffle_with_seed" when training with data augmentation applied is intended.
        Should be set to 0 or 5. By default set to 0.

    Returns:
        None
    """
    # Training parameters
    batch_size = 32
    target_size = (224, 224)
    # input_shape = (224, 224, 3)
    # num_classes = 4
    k = 5
    current_approach = current_approach
    offset = offset_val
    augment_prefixes_list = ["hf_", "co_"]

    adjust_folds_assignment_file(folds_assignment_path, model_free_folds_assignment_path, prefixes_list=augment_prefixes_list)

    for fold in range(offset, k+offset):
        print(f"Training fold {fold}...")
        dg = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_df, train_df = split_data_test_train(folds_assignment_path, folds_assignment_path, offset)
        directory = images_path
        if current_approach == "model_free_shuffle_with_seed":
            test_df, train_df = split_data_test_train(model_free_folds_assignment_path, folds_assignment_path, fold, offset)
            directory = model_free_images_path
        elif current_approach != "basic_shuffle_with_seed":
            print('Invalid approach name given')

        train_gen = dg.flow_from_dataframe(
            dataframe=train_df,
            directory=directory,
            x_col='filename',
            y_col='label',
            target_size=target_size,
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True,
            seed=seed
        )

        test_gen = dg.flow_from_dataframe(
            dataframe=test_df,
            directory=images_path,
            x_col='filename',
            y_col='label',
            target_size=target_size,
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=False,
            seed=seed
        )

        model = prepare_vgg16_model()

        # model training
        os.makedirs('../model_weights', exist_ok=True)
        history = model.fit(train_gen, epochs=5, verbose=True)
        model.save(rf'../model_weights/{current_approach}_vgg16_fold_{fold}.keras')

        # os.mkdir('deep_learning_scores')

        # # training statistics
        np.save(f'scores/deep_learning_scores/{current_approach}_fold{fold}_training_history.npy', history.history)
        precision = np.array(history.history['precision'])
        recall = np.array(history.history['recall'])
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        np.save(f'scores/deep_learning_scores/{current_approach}_fold{fold}_f1.npy', f1)

        # model evaluation
        loss, accuracy, precision, recall = model.evaluate(test_gen, verbose=1)
        print(f"Loss: {loss:.4f}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        eval_metrics = np.array([loss, accuracy, precision, recall])
        np.save(f'scores/deep_learning_scores/{current_approach}_fold{fold}_test_metrics.npy', eval_metrics)

        # using model for prediction on test data
        predicted_types = model.predict(test_gen)
        y_pred = np.argmax(predicted_types, axis=1)
        true_types = test_gen.classes

        # saving true and predicted labels
        np.save(f'scores/deep_learning_scores/{current_approach}_fold{fold}_y_pred.npy', y_pred)
        np.save(f'scores/deep_learning_scores/{current_approach}_fold{fold}_y_true.npy', true_types)

        # saving classification statistics
        report_dict = classification_report(true_types, y_pred, target_names=['fall', 'spring', 'summer', 'winter'],
                                            output_dict=True)
        np.save(f'scores/deep_learning_scores/{current_approach}_fold{fold}_prediction_report.npy', report_dict, allow_pickle=True)