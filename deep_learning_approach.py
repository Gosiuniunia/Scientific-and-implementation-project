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

"""

This script:

- performs a transfer learning of CNN VGG16 model, using provided image data, so it can perform seasonal type classification
- collects model training, evaluation and classification statistics lik 

- Kfolds validation is used in the learning process

"""

def split_data_test_train(k=5):
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
    df = pd.read_csv(FOLDS_ASSIGNMENT_PATH)
    df['filename'] = df['label'] + '/' + df['filename']
    train_df = df[df['kfold'] != fold]
    test_df = df[df['kfold'] == fold]

    return test_df, train_df

def prepare_vgg16_model():
    """

    Returns:

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


IMAGES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\dataset_PColA"
MODEL_FREE_AUGMENTED_IMAGES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\dataset_PCoIA_model_free_augmented"
MODEL_BASED_AUGMENTED_IMAGES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\dataset_PCoIA_model_based_augmented"
FOLDS_ASSIGNMENT_PATH = rf"data/fold_assignments.csv"

# Parameters
batch_size = 32
target_size = (224, 224)
input_shape = (224, 224, 3)
num_classes = 4
k = 5

for fold in range(k):
    print(f"Training fold {fold}...")

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_df, train_df = split_data_test_train(k)

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=IMAGES_PATH,
        x_col='filename',
        y_col='label',
        target_size=target_size,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )

    test_gen = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=IMAGES_PATH,
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
    model.save(rf'C:\Users\wdomc\Documents\personal_color_analysis\model_weights\basic_vgg16_fold_{fold}.keras')

    # # training statistics
    np.save(f'scores/fold{fold}_training_history.npy', history.history)
    precision = np.array(history.history['precision'])
    recall = np.array(history.history['recall'])
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    np.save(f'scores/fold{fold}_f1.npy', f1)

    # Model evaluation
    loss, accuracy, precision, recall = model.evaluate(test_gen, verbose=1)
    print(f"Loss: {loss:.4f}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    eval_metrics = np.array([loss, accuracy, precision, recall])
    np.save(f'scores/fold{fold}_test_metrics.npy', eval_metrics)

    # Using model for prediction on test data
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    # Saving classification results
    np.save(f'scores/fold{fold}_y_pred.npy', y_pred)
    np.save(f'scores/fold{fold}_y_true.npy', y_true)

    # Saving classification statistics
    report_dict = classification_report(y_true, y_pred, target_names=['fall', 'spring', 'summer', 'winter'],
                                        output_dict=True)
    print(report_dict)

    np.save(f'scores/fold{fold}_prediction_report.npy', report_dict, allow_pickle=True)



