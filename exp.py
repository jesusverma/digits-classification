# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import   preprocess_data, split_data, train_model, read_digits, split_train_dev_test, predict_and_eval

###############################################################################


#1. Get the dataset
X,y = read_digits()


_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, X,y):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


# 3. Data Preprocessing
# X_train, X_test, y_train, y_test = split_data(X,y, test_size=0.3)

# train_ratio = 0.75
# dev_size = 0.15
# test_size = 0.10

# # train is now 75% of the entire data set
# x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio)

# # test is now 10% of the initial data set
# # validation is now 15% of the initial data set
# x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_size/(test_ratio + validation_ratio)) 
dev_size = 0.15
test_size = 0.10
X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X,y, dev_size=0.15 , test_size=0.20)


# 4. Data splittting for create train and test sets
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)


#5. Model trainging
model =  train_model(X_train, y_train, {'gamma':0.001}, model_type="svm")


# 6. Finding Model predictions on test set
# Predict the value of the digit on the test subset
predicted = predict_and_eval(model,X_test,y_test) 

###############################################################################
# 7. Qualitative sanity check of the predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")


###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

###############################################################################
# If the results from evaluating a classifier are stored in the form of a
# :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# as follows:


# The ground truth and predicted lists
y_true = []
y_pred = []
cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]

print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)
