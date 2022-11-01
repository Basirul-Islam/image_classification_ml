
import joblib
import numpy as np
import seaborn
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import lbp as LBP

def get_lbp_reports():
    seed = 77
    height, width = 300, 400
    categories = ['Boot', 'Shoe', 'Sandal']  # 'Bacterial leaf blight','Brown spot','Leaf smut'
    # read data
    data, labels = LBP.read_data('DataSet/', categories, height, width)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=seed)

    X_train_feature, X_test_feature = LBP.get_histogram_feature(X_train, X_test)
    lbp = joblib.load('saved_models/lbp.pkl')
    y_pred = lbp.predict(X_test_feature)
    print("The predicted Data is :")
    print(y_pred)
    print("The actual data is:")
    print(np.array(y_test))
    print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")

    test_data, test_labels = LBP.read_data('TestData', categories, height, width)

    features = []

    for sample in test_data:
        features.append(LBP.get_hist_from_lbph(sample))
    y_pred = lbp.predict(features)

    cm = confusion_matrix(test_labels, y_pred)

    print("The predicted Data is :")
    print(y_pred)
    print("The actual data is:")
    print(np.array(test_labels))
    print(f"The model is {accuracy_score(y_pred, test_labels) * 100}% accurate")

    import matplotlib
    matplotlib.use('TkAgg')

    names = ['Boot', 'Shoe', 'Sandal']
    confusion_df = pd.DataFrame(cm, index=names, columns=names)
    plt.figure(figsize=(5, 5))
    seaborn.heatmap(confusion_df, annot=True, annot_kws={"size": 12}, cmap='YlGnBu', cbar=False, square=True, fmt='.2f')
    plt.ylabel(r'True Value', fontsize=14)
    plt.xlabel(r'Predicted Value', fontsize=14)
    plt.tick_params(labelsize=12)
    plt.show()

