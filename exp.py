# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>

# Standard scientific Python imports
import matplotlib.pyplot as plt
import itertools

# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm
from utils import  preprocess_data, split_data, train_model, read_digits, split_train_dev_test, predict_and_eval, tune_hparams, resize_images

###############################################################################

gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
c_ranges = [0.1,1,2,5,10]

# #1. Get the dataset
# X,y = read_digits()


# Load the digits dataset
digits = datasets.load_digits()

# Define image sizes to test
image_sizes = [(4, 4), (6, 6), (8, 8)]

# Define train, dev, and test sizes
train_size = 0.7
dev_size = 0.1
test_size = 0.2
digits = datasets.load_digits()

for size in image_sizes:
    # Resize images to the specified size
    resized_images = [resize(image, size, anti_aliasing=True) for image in digits.images]

    # Flatten the resized images
    n_samples = len(resized_images)
    data = [image.reshape(-1) for image in resized_images]
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(data,digits.target, dev_size , test_size)

    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    X_dev = preprocess_data(X_dev)
        comb_of_gamma_and_c_ranges = [{"gamma_range": x[0], "c_range": x[1]} for x in itertools.product(gamma_ranges,c_ranges)]
        best_model_so_far,best_accuracy_so_far,optimal_gamma,optimal_c = tune_hparams(X_train, y_train,X_dev, y_dev,comb_of_gamma_and_c_ranges)
        test_accuracy = predict_and_eval(best_model_so_far,X_test,y_test) 
        train_accuracy = predict_and_eval(best_model_so_far,X_train,y_train) 
        # print("test_size= ",test_size, ",dev_size= ",dev_size, ",train_size= ", 1-(test_size+dev_size), "train_acc= ", train_accuracy, "dev_acc= ", best_accuracy_so_far, "test_accuracy= ",test_accuracy)
        # print("Best_hparams: gamma", optimal_gamma, "C: ", optimal_c)
        print("image_size:",X ,'x' ,"test_size= ",test_size, ",dev_size= ",dev_size, ",train_size= ", 1-(test_size+dev_size), train_accuracy, "dev_acc= ", best_accuracy_so_far, "test_accuracy= ",test_accuracy)




# print(X1.shape[1],y1)
# print(X2.shape[1],y2)
# print(X3.shape[1],y3)


# Data Preprocessing
# X_train, X_test, y_train, y_test = split_data(X,y, test_size=0.3)

# train_ratio = 0.75
# dev_size = 0.15
# test_size = 0.10

# # train is now 75% of the entire data set
# x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio)

# # test is now 10% of the initial data set
# # validation is now 15% of the initial data set
# x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_size/(test_ratio + validation_ratio)) 
# dev_size = 0.15
# test_size = 0.10
# X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X,y, dev_size=0.15 , test_size=0.20)


#  Data splittting for create train and test sets
# X_train = preprocess_data(X_train)
# X_test = preprocess_data(X_test)
# X_dev = preprocess_data(X_dev)

# Find Hpyer Parameter Tunning here
# For HPT we need to take all combinations of gamma and C


# print("Best_model: ", best_model_so_far)
# print("Best_accuracy: ", best_accuracy_so_far)
# print("Best_hparams: gamma", optimal_gamma, "C: ", optimal_c)



# test_sizes =  [0.2]
# dev_sizes =  [0.1]

# for test_size in test_sizes:
#     for dev_size in dev_sizes:
        
#         X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X,y, dev_size , test_size)

#         X_train = preprocess_data(X_train)
#         X_test = preprocess_data(X_test)
#         X_dev = preprocess_data(X_dev)

#         comb_of_gamma_and_c_ranges = [{"gamma_range": x[0], "c_range": x[1]} for x in itertools.product(gamma_ranges,c_ranges)]
#         best_model_so_far,best_accuracy_so_far,optimal_gamma,optimal_c = tune_hparams(X_train, y_train,X_dev, y_dev,comb_of_gamma_and_c_ranges)
#         test_accuracy = predict_and_eval(best_model_so_far,X_test,y_test) 
#         train_accuracy = predict_and_eval(best_model_so_far,X_train,y_train) 
#         print("test_size= ",test_size, ",dev_size= ",dev_size, ",train_size= ", 1-(test_size+dev_size), "train_acc= ", train_accuracy, "dev_acc= ", best_accuracy_so_far, "test_accuracy= ",test_accuracy)
#         print("Best_hparams: gamma", optimal_gamma, "C: ", optimal_c)

# X_samples =  [X1,X2,X3]
# y_samples =  [y1,y2,y3]

# dev_size = 0.1
# test_size = 0.2
# for idx, X in enumerate(X_samples):
        X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X,y_samples[idx], dev_size , test_size)

        X_train = preprocess_data(X_train)
        X_test = preprocess_data(X_test)
        X_dev = preprocess_data(X_dev)

        comb_of_gamma_and_c_ranges = [{"gamma_range": x[0], "c_range": x[1]} for x in itertools.product(gamma_ranges,c_ranges)]
        best_model_so_far,best_accuracy_so_far,optimal_gamma,optimal_c = tune_hparams(X_train, y_train,X_dev, y_dev,comb_of_gamma_and_c_ranges)
        test_accuracy = predict_and_eval(best_model_so_far,X_test,y_test) 
        train_accuracy = predict_and_eval(best_model_so_far,X_train,y_train) 
        # print("test_size= ",test_size, ",dev_size= ",dev_size, ",train_size= ", 1-(test_size+dev_size), "train_acc= ", train_accuracy, "dev_acc= ", best_accuracy_so_far, "test_accuracy= ",test_accuracy)
        # print("Best_hparams: gamma", optimal_gamma, "C: ", optimal_c)
        print("image_size:",X ,'x' ,"test_size= ",test_size, ",dev_size= ",dev_size, ",train_size= ", 1-(test_size+dev_size), train_accuracy, "dev_acc= ", best_accuracy_so_far, "test_accuracy= ",test_accuracy)




# 3. Vary the test_size in [0.1, 0.2, 0.3] and dev_size in [0.1, 0.2, 0.3], and github actions should output something like: 
#                           test_size=0.1 dev_size=0.1 train_size 0.8 train_acc=xx dev_acc=yy test_acc==zz 
#                                                           {similarly for all 3x3 = 9 combinations}

# 4. For each of these combinations, print the best_hparams found in github actions. 




# Finding Model predictions on test set
# Predict the value of the digit on the test subset
# test_accuracy = predict_and_eval(best_model_so_far,X_test,y_test) 
# print("Testt accuracy: ", test_accuracy)























###############################################################################
# # 7. Qualitative sanity check of the predictions
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")


# ###############################################################################
# # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# # true digit values and the predicted digit values.

# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")

# plt.show()

# ###############################################################################
# # If the results from evaluating a classifier are stored in the form of a
# # :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# # `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# # as follows:


# # The ground truth and predicted lists
# y_true = []
# y_pred = []
# cm = disp.confusion_matrix

# # For each cell in the confusion matrix, add the corresponding ground truths
# # and predictions to the lists
# for gt in range(len(cm)):
#     for pred in range(len(cm)):
#         y_true += [gt] * cm[gt][pred]
#         y_pred += [pred] * cm[gt][pred]

# print(
#     "Classification report rebuilt from confusion matrix:\n"
#     f"{metrics.classification_report(y_true, y_pred)}\n"
# )
