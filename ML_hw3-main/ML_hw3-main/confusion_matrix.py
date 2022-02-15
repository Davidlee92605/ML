#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# In[2]:


y_pred_path = 'x_pred.csv'
y_path = 'train/train/label.csv'
df_y = pd.read_csv(y_path)
df_y = df_y.drop(['id'] , axis = 1)
y_true = df_y.values

df_y_pred = pd.read_csv(y_pred_path)
df_y_pred = df_y_pred.drop(['id'] , axis = 1)
y_pred = df_y_pred.values
class_names = np.array(['Angry' , 'Disgust' , 'Fear' , 'Happy' , 'Sad' , 'Surprise' , 'Neutral'])


# In[9]:


def plot_confusion_matrix(y_true , y_pred , classes , normalize = False , title = None , cmap = plt.cm.Blues):
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots(figsize=(12,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
    


# In[11]:


plot_confusion_matrix(y_true , y_pred , classes=class_names , title='Confusion matrix')
plt.show()


# In[ ]:




