# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import BatchNormalization, LeakyReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp

import time

import seaborn as sn

# This sample preprocessed dataset contains only 10000 records. 
# Download the data from path mentioned in README
# The original datset contains 2.5 million records and 48 attributes
dataset = pd.read_csv('dataset_final.csv')

dataset.shape

dataset[0:5]

X = dataset.iloc[:,0:36]
X.shape

# Last two columns are output columns
y = dataset.iloc[:,-2:]
y.shape

y_attack_cat = dataset.iloc[:,-2]
y_attack_cat.shape

def plot_attack_category(dataset, yattack):
    sn.set(rc={'figure.figsize':(15,10),"font.size":70,"axes.titlesize":40,"axes.labelsize":20},style="white")
    catg_plot = sn.countplot(yattack,data = dataset)
    for p in catg_plot.patches:
      height = p.get_height()
      catg_plot.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center",fontsize = 15)
    catg_plot.figure.savefig('attack_cats.png')
    catg_plot.figure.show()

plot_attack_category(dataset,y_attack_cat)

from scipy import stats
from sklearn.preprocessing import MinMaxScaler
# Pass parameter type = 'statistical' or 'minmax'
# Note: statistical is better for SVM and KNN.
def normalization(data, type='minmax'):
    if type == 'statistical':
        return stats.zscore(data)
    elif type == 'minmax':
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)
    else:
        print('\n Norm type not found! \n')

X = normalization(X, 'statistical')

def plot_roc(y_pred_val, y_testlb_val, y_predlb_val, classifier,count):
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(len(np.unique(y_pred_val))):
    fpr[i], tpr[i], _ = roc_curve(y_testlb_val[:, i], y_predlb_val[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
  n_classes = len(np.unique(y_pred_val))
  print('n_classes',n_classes)
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Plot all ROC curves
  plt.figure()
  colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
  for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
               label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))
  plt.plot([0, 1], [0, 1], 'k--', lw=2)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristics '+classifier)
  plt.legend(loc="lower right")
  plt.savefig('ROC '+classifier+' '+str(count)+'.png')
  plt.show()
  plt.clf()

# Stratified K-Fold cross validation
skf = StratifiedKFold(n_splits=5,shuffle = True)

"""Below is the implementation of six different models - Naive Bayes, Adaboost, XGBoost, KNN, Logistic Regression and ANN.
The evaluation metrics used are F1 Score, ROC, Confusion matrix, Loss vs Accuracy (for ANN)
"""

from sklearn.naive_bayes import GaussianNB
time_NB =[]
f1score_NB = []

count = 1
# Training Model with Naive Bayes :
for train_index, test_index in skf.split(X, y_attack_cat):
  
  print("TRAIN:", train_index, "TEST:", test_index)

  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y_attack_cat[train_index], y_attack_cat[test_index]
  
  s = time.clock()
  NB_model = GaussianNB()
  NB_model.fit(X_train,y_train)
  y_pred = NB_model.predict(X_test)
  print('ypred unique \n',np.unique(y_pred))
  lb = preprocessing.LabelBinarizer()
  y_testlb = lb.fit_transform(y_test)
  y_predlb = lb.fit_transform(y_pred)
  print('shape pred \n',y_predlb.shape)
  print('shape test \n',y_testlb.shape)

  e = time.clock() - s
  time_NB.append(e) 
  score = f1_score(y_test,y_pred, average='micro')
  f1score_NB.append(score)
  print("Time taken by NB Gaussian means model: ", e,"seconds \n")
  print("F1 score for iteration: ",score,'\n')

  plot_roc(y_pred, y_testlb, y_predlb, 'Naive Baiyes',count)
  count +=1
  
print("F1 score for NB Gaussian model prediction: ",f1score_NB,'\n')

print('f1score NB: ',f1score_NB)
print('time NB: ',time_NB)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# Training Model with Adaboost Classifier :
ada_score = []
time_ada = []
count =1
for train_index, test_index in skf.split(X, y_attack_cat):
     print("TRAIN:", train_index, "TEST:", test_index)

     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y_attack_cat[train_index], y_attack_cat[test_index]
     
     s = time.clock()


     ada_model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(min_samples_split=4,random_state=21),n_estimators=100)
     ada_model.fit(X_train,y_train)

     y_pred = ada_model.predict(X_test)
     lb = preprocessing.LabelBinarizer()
     y_testlb = lb.fit_transform(y_test)
     y_predlb = lb.fit_transform(y_pred)

     e = time.clock() - s
     time_ada.append(e) 
     score = f1_score(y_test,y_pred, average='micro')
     ada_score.append(score)
     print("Time taken by Ada model: ", e,"seconds \n")
     print("F1 score at iterartion : ",score,'\n')
     plot_roc(y_pred, y_testlb, y_predlb, 'Adaboost',count)
     count +=1

print("F1 score for Adaboost model prediction: ",ada_score,'\n')
print("Time for Adaboost model prediction: ",time_ada,'\n')

from xgboost import XGBClassifier
score_roc_xg = []
score_f1_xg = []
xg_time = []
y_pred_xg_all = []
y_test_xg_all = []

count =1
for train_index, test_index in skf.split(X, y_attack_cat):
     print("TRAIN:", train_index, "TEST:", test_index)

     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y_attack_cat[train_index], y_attack_cat[test_index]
     
     s = time.clock()

     XGB_model = XGBClassifier(learning_rate =0.1, n_estimators=100, num_class = 12,
                             min_child_weight=1, gamma=0,subsample=0.8,colsample_bytree=0.8,
                             objective= 'multi:softmax', nthread=4,scale_pos_weight=1,
                             seed=27, early_stopping_rounds=70, verbose=False)

     XGB_model.fit(X_train,y_train)

     y_pred = XGB_model.predict(X_test)
     lb = preprocessing.LabelBinarizer()
     y_testlb = lb.fit_transform(y_test)
     y_predlb = lb.fit_transform(y_pred)

     e = time.clock() - s
     xg_time.append(e)

     y_pred_xg_all.append(y_pred)
     y_test_xg_all.append(y_test)

     plot_roc(y_pred, y_testlb, y_predlb, 'XGBoost',count)
     count +=1

     roc_score = roc_auc_score(y_testlb,y_predlb)
     print('ROC SCORE:\n',roc_score)   
     score_roc_xg.append(roc_score)
     f1score = f1_score(y_testlb, y_predlb, average='micro')
     print('F1 SCORE:\n',f1score)
     #print('CONFUSION MATRIX:\n',confusion_matrix(y_testlb, y_predlb))
     score_f1_xg.append(f1score)
print("F1 score for XGB model prediction: ",score_f1_xg,'\n')
print("Time for XGB model prediction: ",xg_time,'\n')

cv_score_roc = []
cv_score_f1 = []

for train_index, test_index in skf.split(X, y_attack_cat):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_attack_cat[train_index], y_attack_cat[test_index]
     
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train,y_train)
    y_pred = knn_model.predict(X_test)

    lb = preprocessing.LabelBinarizer()
    y_testlb = lb.fit_transform(y_test)
    y_predlb = lb.fit_transform(y_pred)

    roc_score = roc_auc_score(y_testlb,y_predlb)
    print('ROC SCORE:\n',roc_score)   
    cv_score_roc.append(roc_score)
    f1score = f1_score(y_testlb, y_predlb, average='micro')
    print('F1 SCORE:\n',f1score)
    #print('CONFUSION MATRIX:\n',confusion_matrix(y_testlb, y_predlb))
    cv_score_f1.append(f1score)
print("F1 score for KNN model prediction: ",cv_score_f1,'\n')

from sklearn.linear_model import LogisticRegression
time_log = []
cv_score_roc_log = []
cv_score_f1_log = []
y_pred_log_all = []
y_test_log_all = []

count = 1
for train_index, test_index in skf.split(X, y_attack_cat):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_attack_cat[train_index], y_attack_cat[test_index]
    
    s = time.clock()
    
    lr_model = LogisticRegression(random_state = 0, max_iter = 10000, solver = 'lbfgs')
    lr_model.fit(X_train,y_train)
    y_pred = lr_model.predict(X_test)

    lb = preprocessing.LabelBinarizer()
    y_testlb = lb.fit_transform(y_test)
    y_predlb = lb.fit_transform(y_pred)
    
    e = time.clock() - s
    time_log.append(e)
    
    y_pred_log_all.append(y_pred)
    y_test_log_all.append(y_test)
    
    plot_roc(y_pred, y_testlb, y_predlb, 'Logistic Regression',count)
    count +=1

    roc_score = roc_auc_score(y_testlb,y_predlb)
    print('ROC SCORE:\n',roc_score)   
    cv_score_roc_log.append(roc_score)
    f1score = f1_score(y_testlb, y_predlb, average='micro')
    print('F1 SCORE:\n',f1score)
    #print('CONFUSION MATRIX:\n',confusion_matrix(y_testlb, y_predlb))
    cv_score_f1_log.append(f1score)
print("F1 score for Log Reg model prediction: ",cv_score_f1_log,'\n')
print("Time for Log Reg model prediction: ",time_log,'\n')

model = Sequential()
model.add(Dense(256, input_shape=(36,)))
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(0.2))
model.add(Dropout(.4))
model.add(Dense(128))
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(0.2))
model.add(Dropout(.4))
model.add(Dense(64))
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(0.2))
model.add(Dropout(.4))
model.add(Dense(16))
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(0.2))
model.add(Dropout(.4))
model.add(Dense(10, activation='softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

time_ANN = []

cv_score_ann = []
all_history = []
y_pred_ann_all = []
y_test_ann_all = []
for train_index, test_index in skf.split(X, y_attack_cat):
     print("TRAIN:", train_index, "TEST:", test_index)

     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y_attack_cat[train_index], y_attack_cat[test_index]

     s = time.clock()

     es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 5, 
                        restore_best_weights = True, verbose = 1)
     rlrop = ReduceLROnPlateau(monitor = 'val_loss', mode = 'min', patience = 5, 
                               factor = 0.2, min_lr = 1e-6, verbose = 1)

     history = model.fit(X_train, y_train, epochs=50, batch_size=512,
               validation_split=0.2, verbose=1,
               callbacks = [es, rlrop],
               shuffle=True) 
     score1 = model.evaluate(X_test, y_test)
     print('SCORE1:\n', score1)
     y_pred = model.predict(X_test)

     e = time.clock() - s
     time_ANN.append(e)
     print("Time taken by ANN model: ", e,"seconds \n")
     cv_score_ann.append(score1)
     all_history.append(history)
     y_pred_ann_all.append(y_pred)
     y_test_ann_all.append(y_test)

y_pred_class = []
for i in range(5):
  temp = []
  for j in range(len(y_pred_ann_all[i])):
    #temp.append(list(y_pred_all[i]).index(max(y_pred_all[i][j])).any())
    temp.append(y_pred_ann_all[i][j].argmax(axis=0))
  y_pred_class.append(temp)

for i in range(5):
  print(f1_score(y_test_ann_all[i], y_pred_class[i], average='micro'))

for i in range(5):
  print("Run ", i)
  print(confusion_matrix(y_test_ann_all[i], y_pred_class[i]))

def plot_confusion_matrix(cm, classes,
    normalize=False,
    title='Confusion matrix',
    cmap=plt.cm.Blues):
    sn.set(font_scale=1.0)
    sn.heatmap(cm, cmap="BuPu", annot=True,cbar=False,fmt='d')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion.png')


cmatrix = confusion_matrix(y_test_ann_all[4], y_pred_class[4], labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) #5th iter has best Confusion Matrix
cm = pd.DataFrame(cmatrix, range(10), range(10))
plot_confusion_matrix(cm, ['Normal', 'Exploits', 'Reconnaissance', 'DoS', 'Generic', 'Shellcode', 'Fuzzers', 'Worms', 'Backdoors', 'Analysis'], normalize=False)

i=1
for hist in all_history:
  train_loss = hist.history['loss']
  val_loss = hist.history['val_loss']
  train_acc = hist.history['accuracy']
  val_acc = hist.history['val_accuracy']
  xc = range(len(train_loss))
  plt.figure(1, figsize=(7, 5))
  plt.plot(xc, train_loss)
  plt.plot(xc, val_loss)
  plt.xlabel('num of Epochs')
  plt.ylabel('loss')
  plt.title('Loss:Train vs Val')
  plt.grid(True)
  plt.legend(['train', 'val'])
  plt.style.use(['classic'])
  plt.savefig('Loss_Plot'+str(i)+'.png')
  plt.show()
  plt.clf()

  plt.figure(2, figsize=(7, 5))
  plt.plot(xc, train_acc)
  plt.plot(xc, val_acc)
  plt.xlabel('num of Epochs')
  plt.ylabel('accuracy')
  plt.title('Accuracy:Train vs Val')
  plt.grid(True)
  plt.legend(['train', 'val'], loc=4)
  plt.style.use(['classic'])
  plt.savefig('Accuracy_Plot'+str(i)+'.png')
  plt.show()
  plt.clf()
  i+=1