
import numpy as np
import sys, os
import pickle 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque
import cv2

# My
from .feature_proc import FeatureGenerator
from .funcs import int2str

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA

# Path
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

# Feiyu's for Inference -----------------------------------------------
class MyClassifier(object):
    
    def __init__(self, model_path, action_types):
        
        # -- Settings
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        if self.model is None:
            print("my Error: failed to load model")
            assert False
        self.action_types = action_types
        self.THRESHOLD_MIN_SCORE = 0.7

        # -- Time serials storage
        self.cnt = 0
        self.fg = FeatureGenerator()
        self.reset()
        
    def reset(self):
        self.fg.reset()
        self.scores_hist = deque()
        self.scores = None

    def insert_and_get_score(self, curr_score):
        self.scores_hist.append(curr_score)
        DEQUE_MAX_SIZE = 2
        if len(self.scores_hist)>DEQUE_MAX_SIZE:
            self.scores_hist.popleft()
        if 1: # Use sum
            score_sums = np.zeros((len(self.action_types),))
            for score in self.scores_hist:
                score_sums += score
            score_sums /= len(self.scores_hist)
            print("\nMean score:\n", score_sums)
            return score_sums

        else: # Use multiply
            score_mul = np.ones((len(self.action_types),))
            for score in self.scores_hist:
                score_mul *= score
            return score_mul

    def predict(self, skeleton):
        LABEL_UNKNOWN = ""
        flag, features = self.fg.add_curr_skeleton(skeleton)

        if flag:
            features = features.reshape(-1, features.shape[0]) # convert to 2d array

            curr_score = self.model.predict_proba(features)[0]

            self.scores = self.insert_and_get_score(curr_score)

            if self.scores.max() < self.THRESHOLD_MIN_SCORE:
                prediced_label = LABEL_UNKNOWN
            else:
                # predicted_idx = scores.index(max(scores))
                predicted_idx = self.scores.argmax()
                prediced_label = self.action_types[predicted_idx]
        else:
            prediced_label = LABEL_UNKNOWN
        return prediced_label

    def draw_scores_onto_image(self, image_disp):
        if self.scores is None:
            return
        for i, label in enumerate( self.action_types ):
            txt_x = 20
            txt_y = 60 + i*30
            cv2.putText(image_disp,
                "{:<9}: {:.2f}".format(label, self.scores[i]),
                (txt_x, txt_y),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)

# Define classifier for training-----------------------------------------------
class MyModel(object):
    def __init__(self):
        self.init_all_models()

        # self.clf = self.choose_model("Nearest Neighbors")
        # self.clf = self.choose_model("Linear SVM")
        # self.clf = self.choose_model("RBF SVM")
#         self.clf = self.choose_model("Gaussian Process")
        # self.clf = self.choose_model("Decision Tree")
        # self.clf = self.choose_model("Random Forest")
        # self.clf = MLPClassifier((200,200,200,200,200,200)) # 99, 97
        # self.clf = MLPClassifier((200, 200, 150, 150, 100, 100))
        # self.clf = MLPClassifier((100, 100, 100, 100, 100))
        self.clf = MLPClassifier((100, 100, 100))
        # self.clf = MLPClassifier((50, 50, 50))
        # self.clf = MLPClassifier((80, 80, 80, 80))
        # self.clf = self.choose_model("AdaBoost")
#         self.clf = self.choose_model("Naive Bayes")
        # self.clf = self.choose_model("QDA")
        

    def choose_model(self, name):
        idx = self.names.index(name)
        return self.classifiers[idx]
            
    def init_all_models(self):
        self.names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

        self.classifiers = [
            KNeighborsClassifier(5),
            SVC(kernel="linear", C=10.0),
            SVC(gamma=0.01, C=1.0, verbose=True),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=30, n_estimators=100, max_features="auto"),
            MLPClassifier(alpha=1),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]
    
    def train(self, X, Y):
        n_components = min(50, X.shape[1])
        self.pca = PCA(n_components=n_components, whiten=True)
        self.pca.fit(X)
        # print("Sum eig values:", np.sum(self.pca.singular_values_))
        print("Sum eig values:", np.sum(self.pca.explained_variance_ratio_))
        X_new = self.pca.transform(X)
        print("After PCA, X.shape = ", X_new.shape)

        self.clf.fit(X_new, Y)
        
    def predict(self, X):
        Y_predict = self.clf.predict(self.pca.transform(X))
        return Y_predict

    def predict_proba(self, X):
        Y_probs = self.clf.predict_proba(self.pca.transform(X))
        return Y_probs

    def predict_and_evaluate(self, te_X, te_Y):
        te_Y_predict = self.predict(te_X)
        N = len(te_Y)
        n = sum( te_Y_predict == te_Y )
        accu = n / N
        print("Accuracy is ", accu)
#         print(te_Y_predict)
        return accu, te_Y_predict