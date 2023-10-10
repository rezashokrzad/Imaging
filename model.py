# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:27:41 2023

@author: rezas
"""

import tensorflow as tf
import matplotlib.pyplot as plt # visualization
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# load the dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# reshape dataset
X_train = train_images.reshape(train_images.shape[0], 28*28)
X_test = test_images.reshape(test_images.shape[0], 28*28)
y_train = train_labels
y_test = test_labels

# Plotting 6 sample images with their labels
fig, axes = plt.subplots(2, 3, figsize=(10, 7))

for ax, image, label in zip(axes.ravel(), train_images, train_labels):
    ax.imshow(image, cmap=plt.cm.gray_r)
    ax.set_title(f"Label: {label}")
    ax.axis('off')

plt.tight_layout()
plt.show()


# save some samples for later tests
image = test_images[142]
plt.imsave('test4.png', image, cmap=plt.cm.gray)


# train pahse
clf = RandomForestClassifier(n_estimators=200)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_pred, y_test))

# Save the trained model
joblib.dump(clf, 'random_forest_mnist.pkl')

































