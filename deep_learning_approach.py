import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import Precision, Recall
import numpy as np
from sklearn.metrics import classification_report
import csv

"""

This script:

- performs a transfer learning of CNN VGG16 model, using provided image data, so it can perform seasonal type classification
- collects model training, evaluation and classification statistics lik 

- Kfolds validation is used in the learning process

"""

def split_data_test_train(assignment_file_path, k=5):
    """
    Splits images details (file path, label),  provided as pandas Dataframe, into train and test sets details pandas Dataframes,
    taking into account the fold number k
    Args:
        k: number of current fold

    Returns:
        train_df: train data details pandas Dataframe, including label and file path
        test_df:  test data details pandas Dataframe, including label and file path

    """
    # Retrieving and adjusting folds assignments details
    df = pd.read_csv(assignment_file_path)
    df['filename'] = df['label'] + '/' + df['filename']

    train_df = df[df['kfold'] != fold]
    test_df = df[df['kfold'] == fold]

    return test_df, train_df

def prepare_vgg16_model():
    """

    Function sets up a VGG16 model with Imagenet pre-trained weights for 4 classes classification task.
    Model reference: https://keras.io/api/applications/vgg/

    Returns:
        model: VGG16 model

    """
    # model definition
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
        assignment_file_path: path to a .csv file with folds assignment information
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
            for prefix in prefixes_list:
                new_row = row.copy()
                new_row["filename"] = prefix + original_filename
                new_rows.append(new_row)

    with open(assignment_file_output_path, mode='w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(new_rows)


IMAGES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\dataset_PColA"
MODEL_FREE_AUGMENTED_IMAGES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\dataset_PCoIA_model_free_augmented"
MODEL_BASED_AUGMENTED_IMAGES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\dataset_PCoIA_model_based_augmented"
FOLDS_ASSIGNMENT_PATH = rf"data/fold_assignments.csv"
MODEL_FREE_FOLDS_ASSIGNMENT_PATH = rf"data/model_free_fold_assignments.csv"
MODEL_BASED_FOLDS_ASSIGNMENT_PATH = rf"data/model_based_fold_assignments.csv"

# Training parameters
batch_size = 32
target_size = (224, 224)
input_shape = (224, 224, 3)
num_classes = 4
k = 5
current_approach = "model_free"
augment_prefixes_list = ["hf_", "co_"]

adjust_folds_assignment_file(FOLDS_ASSIGNMENT_PATH, MODEL_FREE_FOLDS_ASSIGNMENT_PATH, prefixes_list=augment_prefixes_list)

for fold in range(k):
    print(f"Training fold {fold}...")

    dg = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_df, train_df = split_data_test_train(MODEL_FREE_FOLDS_ASSIGNMENT_PATH, k)

    train_gen = dg.flow_from_dataframe(
        dataframe=train_df,
        directory=MODEL_FREE_AUGMENTED_IMAGES_PATH,
        x_col='filename',
        y_col='label',
        target_size=target_size,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )

    test_gen = dg.flow_from_dataframe(
        dataframe=test_df,
        directory=MODEL_FREE_AUGMENTED_IMAGES_PATH,
        x_col='filename',
        y_col='label',
        target_size=target_size,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    model = prepare_vgg16_model()

    # model training
    history = model.fit(train_gen, epochs=5, verbose=True)
    model.save(rf'C:\Users\wdomc\Documents\personal_color_analysis\model_weights\{current_approach}_vgg16_fold_{fold}.keras')

    # # training statistics
    np.save(f'scores/{current_approach}_fold{fold}_training_history.npy', history.history)
    precision = np.array(history.history['precision'])
    recall = np.array(history.history['recall'])
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    np.save(f'scores/{current_approach}_fold{fold}_f1.npy', f1)

    # Model evaluation
    loss, accuracy, precision, recall = model.evaluate(test_gen, verbose=1)
    print(f"Loss: {loss:.4f}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    eval_metrics = np.array([loss, accuracy, precision, recall])
    np.save(f'scores/{current_approach}_fold{fold}_test_metrics.npy', eval_metrics)

    # Using model for prediction on test data
    predicted_types = model.predict(test_gen)
    y_pred = np.argmax(predicted_types, axis=1)
    true_types = test_gen.classes

    # Saving classification results
    np.save(f'scores/{current_approach}_fold{fold}_y_pred.npy', y_pred)
    np.save(f'scores/{current_approach}_fold{fold}_y_true.npy', true_types)

    # Saving classification statistics
    report_dict = classification_report(true_types, y_pred, target_names=['fall', 'spring', 'summer', 'winter'],
                                        output_dict=True)
    print(report_dict)
    np.save(f'scores/{current_approach}_fold{fold}_prediction_report.npy', report_dict, allow_pickle=True)