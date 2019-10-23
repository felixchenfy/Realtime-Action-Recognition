if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.lib_feature_proc as lib_feature_proc


def main():
    # -- Train-test split
    tr_X, te_X, tr_Y, te_Y = lib_feature_proc.train_test_split(X, Y)
    print("\nAfter train-test split:")
    print("Size of training data X:    ", tr_X.shape)
    print("Number of training samples: ", len(tr_Y))
    print("Number of testing samples:  ", len(te_Y))

if __name__ == "__main__":
    main()