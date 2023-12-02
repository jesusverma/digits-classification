# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>

# Standard scientific Python imports
import matplotlib.pyplot as plt
import itertools

# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm

from utils import  preprocess_data, split_data, train_model, read_digits, split_train_dev_test, predict_and_eval, tune_hparams, get_hyperparameter_combinations
import joblib

from joblib import dump, load
import pandas as pd
import os

###############################################################################



#1. Get the dataset
X,y = read_digits()

# 2. Define common Hyper parameter combinations
classifier_param_dict = {}

gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
c_ranges = [0.1,1,2,5,10]


h_params={}
h_params['gamma'] = gamma_ranges
h_params['C'] = c_ranges
h_params_combinations = get_hyperparameter_combinations(h_params)
classifier_param_dict['svm'] = h_params_combinations


#  Decision Tree h params
max_depth_list = [5, 10, 15, 20, 50, 100]
h_params_tree = {}
h_params_tree['max_depth'] = max_depth_list
h_params_trees_combinations = get_hyperparameter_combinations(h_params_tree)
classifier_param_dict['tree'] = h_params_trees_combinations


h_params_lr={}
h_params_lr['solver'] = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
h_params_lr['C'] = [100]
h_params_lr_combinations = get_hyperparameter_combinations(h_params_lr)
classifier_param_dict['logistic_regression'] = h_params_lr_combinations




results = []
test_sizes =  [0.2]
dev_sizes  =  [0.1]


num_runs  = 5

def save_model(model, filename):
    joblib.dump(model, filename)

for cur_run_i in range(num_runs):
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            train_size = 1- test_size - dev_size
            
            X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X,y, dev_size , test_size)

            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            X_dev = preprocess_data(X_dev)

            for model_type in classifier_param_dict:
                model_type_hparams = classifier_param_dict[model_type]

                best_hparams,best_model_path,best_accuracy_so_far  = tune_hparams(X_train, y_train, X_dev, 
                y_dev, model_type_hparams,model_type)  

                print(best_model_path)
                best_model_so_far = load(best_model_path) 

                test_accuracy = predict_and_eval(best_model_so_far,X_test,y_test) 
                train_accuracy = predict_and_eval(best_model_so_far,X_train,y_train) 
                dev_accuracy = best_accuracy_so_far

                print("{}\ttest_size={:.2f} dev_size={:.2f} train_size={:.2f} train_accuracy={:.2f} dev_accuracy={:.2f} test_accuracy={:.2f}".format(model_type, test_size, dev_size, train_size, train_accuracy, dev_accuracy, test_accuracy))
                
                if model_type == "logistic_regression":
                    print("Jesuis",best_model_path)

                cur_run_results = {'model_type': model_type, 'run_index': cur_run_i, 'train_accuracy' : train_accuracy, 'dev_accuracy': dev_accuracy, 'test_accuracy': test_accuracy}
                results.append(cur_run_results)            
        
print(pd.DataFrame(results).groupby('model_type').describe().T)


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
