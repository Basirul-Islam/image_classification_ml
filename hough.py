# -*- coding: utf-8 -*-
"""hough.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dLXf1IAVPRKNOWFik9FiRSuoZQCYNfNC
"""
import seaborn
from skimage.transform import probabilistic_hough_line
from skimage.transform import hough_line
from skimage.feature import canny
import pandas as pd
def hough(image_resized):
  edges = canny(image_resized,sigma = 3)
  # print(edges.shape, np.amax(edges), np.amin(edges))
  lines = probabilistic_hough_line(edges, threshold = 10, line_length = 5, line_gap = 3)
  fd = []
  
  for line in lines[:20]:
    # f = []
    p1, p2 = line
    fd += [p1[0], p1[1], p2[0], p2[1]]
    
  fd += [len(lines), sum(fd)]
  return fd

# import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
from skimage.filters import gaussian
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

Categories = ['Boot', 'Shoe', 'Sandal']


#path which contains all the categories of images
def feature_extraction(datadir):
  data = [] #input array
  labels = [] #output array
  for category in Categories:    
      print(f'Reading... category : {category}')
      count = 0
      path=os.path.join(datadir,category)
      for img in os.listdir(path):
          count = count + 1
          if (count > 10):
              break
          img_array = imread(os.path.join(path,img), as_gray=True)
          img_array = gaussian(img_array, sigma = .4)
          img_resized = resize(img_array,(300, 400))
          img_hough = hough(img_resized)
          # print(img_hog.shape)
          data.append(img_hough)
          labels.append(Categories.index(category))
      print(f'Readed category: {category} successfully')
  return data, labels

"""# Model Construction"""
def get_hough_result():

    from sklearn import svm
    from sklearn.model_selection import GridSearchCV

    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}
    svc = svm.SVC(probability = True)
    model = GridSearchCV(svc, param_grid, error_score = 'raise')

    """# Feature Extraction"""

    data, labels = feature_extraction('DataSet/')

    """# Training-Testing Split"""

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.20, random_state = 77, shuffle = True, stratify = labels)
    # print(x_train)
    print('Splitted Successfully')

    # model.best_params_ contains the best parameters obtained from GridSearchCV

    """# Model Fit"""

    model.fit(x_train, y_train)
    print('The Model is trained well with the given images')

    """# Model Testing"""

    from sklearn.metrics import classification_report,accuracy_score
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_pred = model.predict(x_train)
    # y_true = y_pred
    cm = confusion_matrix(y_train, y_pred)

    print("The predicted Data is :")
    print(y_pred)
    print("The actual data is:")
    print(np.array(y_train))
    print(f"The model is {accuracy_score(y_pred,y_train)*100}% accurate")
    import matplotlib
    matplotlib.use('TkAgg')

    names = ['Boot', 'Shoe', 'Sandal']
    confusion_df = pd.DataFrame(cm, index=names, columns=names)
    plt.figure(figsize=(5, 5))
    seaborn.heatmap(confusion_df, annot=True, annot_kws={"size": 12}, cmap='YlGnBu', cbar=False, square=True, fmt='.2f')
    plt.ylabel(r'True Value', fontsize=14)
    plt.xlabel(r'Predicted Value', fontsize=14)
    plt.title('HOUGH')
    plt.tick_params(labelsize=12)
    plt.show()
    # img = imread('/content/drive/MyDrive/animal/test/cow/177bc31f99.jpg', as_gray=True)

    # # plt.show()
    # img_array = gaussian(img, sigma = .4)

    # img_resized = resize(img_array, (300, 400))
    # print(img_resized)
    # print(10000000000000)
    # plt.imshow(img_resized)
    # img_hough = hough(img_resized)
    # img_hough = np.array(img_hough).reshape(1, -1)
    # print(img_hough)

    # probability = model.predict_proba(img_hough)
    # ans = model.predict(img_hough)
    # print(ans)
    # print(probability)
    # maxi = 0
    # res = 'cow'
    # for ind, val in enumerate(Categories):
    #     prob = probability[0][ind]
    #     print(f'{val} = {prob*100}%')
    #     if prob > maxi:
    #       maxi = prob
    #       res = val

    # print(f"The predicted image is : {res}") #+ Categories[model.predict(img_hough)[0]]

    # test_data, test_labels = feature_extraction('TestData')
    #
    # y_pred = model.predict(test_data)
    #
    # cm = confusion_matrix(test_labels, y_pred)
    # #disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = Categories)
    # #disp.plot()
    # print("The predicted Data is :")
    # print(y_pred)
    # print("The actual data is:")
    # print(np.array(test_labels))
    # print(f"The model is {accuracy_score(y_pred, test_labels)*100}% accurate")