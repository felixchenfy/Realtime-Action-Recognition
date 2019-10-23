''' This script does:
1. Load features and classes from csv files
2. Train the model
3. Save the model to `model/` folder.
'''

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.lib_plot as lib_plot
    from utils.lib_feature_proc import train_test_split
    from utils.lib_classifier import ClassifierOfflineTrain

# -- Settings

SRC_CLASSES = ROOT + "data_proc/classes.csv"
SRC_PROCESSED_FEATURES = ROOT + "data_proc/features_X.csv"
SRC_PROCESSED_FEATURES_LABELS = ROOT + "data_proc/features_Y.csv"

DST_MODEL_PATH = ROOT + 'model/trained_classifier_tmp.pickle'

# -- Functions


def load_classes_features_and_labels():
    ''' Load classes.csv, features_X.csv, features_Y.csv.
        The filepath is defined under the `Settings` section.
    '''
    with open(SRC_CLASSES, 'r') as f:
        classes = [str_class.rstrip() for str_class in f if str_class != '\n']
    classes = np.array(classes)
    features_X = np.loadtxt(SRC_PROCESSED_FEATURES, dtype=float)  # features
    features_Y = np.loadtxt(SRC_PROCESSED_FEATURES_LABELS, dtype=int)  # labels
    return classes, features_X, features_Y


def evaluate_model(model, classes, tr_X, tr_Y, te_X, te_Y):
    ''' Evaluate accuracy and time cost '''

    # Accuracy
    t0 = time.time()

    tr_accu, tr_Y_predict = model.predict_and_evaluate(tr_X, tr_Y)
    print(f"Accuracy on training set is {tr_accu}")

    te_accu, te_Y_predict = model.predict_and_evaluate(te_X, te_Y)
    print(f"Accuracy on testing set is {te_accu}")

    print("Accuracy report:")
    print(classification_report(
        te_Y, te_Y_predict, target_names=classes, output_dict=False))

    # Time cost
    average_time = (time.time() - t0) / (len(tr_Y) + len(te_Y))
    print("Time cost for predicting one sample: "
          "{:.3f} seconds".format(average_time))

    # Plot accuracy
    axis, cf = lib_plot.plot_confusion_matrix(
        te_Y, te_Y_predict, classes, normalize=False, size=(12, 8))
    plt.show()



# -- Main


def main():

    # -- Load preprocessed data
    print("\nReading csv files of classes, features, and labels ...")
    classes, X, Y = load_classes_features_and_labels()

    # -- Train-test split
    tr_X, te_X, tr_Y, te_Y = train_test_split(X, Y)
    print("\nAfter train-test split:")
    print("Size of training data X:    ", tr_X.shape)
    print("Number of training samples: ", len(tr_Y))
    print("Number of testing samples:  ", len(te_Y))

    # -- Train the model
    print("\nStart training model ...")
    model = ClassifierOfflineTrain()
    model.train(tr_X, tr_Y)

    # -- Evaluate model
    print("\nStart evaluating model ...")
    evaluate_model(model, classes, tr_X, tr_Y, te_X, te_Y)

    # -- Save model
    print("\nSave model to " + DST_MODEL_PATH)
    with open(DST_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
