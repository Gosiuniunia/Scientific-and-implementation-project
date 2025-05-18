import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import pandas as pd

IMAGES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\dataset_PColA"
FOLDS_ASSIGNMENT_PATH = rf"data/fold_assignments.csv"

# Parameters
batch_size = 32
target_size = (224, 224)
input_shape = (224, 224, 3)
num_classes = 4

df = pd.read_csv(FOLDS_ASSIGNMENT_PATH)
df['filename'] = df['label'] + '/' + df['filename']

k = 1

for fold in range(k):
    print(f"Testing fold {fold}...")

    train_df = df[df['kfold'] != fold]
    test_df = df[df['kfold'] == fold]

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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

    test_model = load_model(rf"C:\Users\wdomc\Documents\personal_color_analysis\model_weights\basic_vgg16_fold_0.keras")
    loss, accuracy, precision, recall = test_model.evaluate(test_gen, verbose=1)
    print(f"Loss: {loss:.4f}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    eval_metrics = np.array([loss, accuracy, precision, recall])
    np.save(f'scores/fold{fold}_test_metrics.npy', eval_metrics)

    # Using model for prediction on test data
    y_pred_probs = test_model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    report_dict = classification_report(y_true, y_pred, target_names=['fall', 'spring', 'summer', 'winter'],
                                        output_dict=True)
    print(report_dict)