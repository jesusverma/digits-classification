
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, svm
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load



def get_combinations(param_name, param_values, base_combinations):    
    new_combinations = []
    for value in param_values:
        for combination in base_combinations:
            combination[param_name] = value
            new_combinations.append(combination.copy())    
    return new_combinations
    
def get_hyperparameter_combinations(dict_of_param_lists):    
    base_combinations = [{}]
    for param_name, param_values in dict_of_param_lists.items():
        base_combinations = get_combinations(param_name, param_values, base_combinations)
    return base_combinations



def read_digits():
    digits =  datasets.load_digits()
    X = digits.images
    y = digits.target
    return X,y

#find all utils here
def preprocess_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into 50% train and 50% test subsets
def split_data(x,y,test_size,random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
        x,y, test_size=test_size,random_state=random_state
    )
    return X_train, X_test, y_train, y_test



def split_train_dev_test(x,y,dev_size,test_size,random_state=1):
    X_train_dev, X_test, y_train_dev, y_test = split_data(
        x,y, test_size=test_size,random_state=random_state
    )
    # test is now (test_size * 100)% of the initial data set
    # dev is now (dev_size * 100)% of the initial data set
    X_train, X_dev, y_train, y_dev = split_data(X_train_dev, y_train_dev, dev_size/(1-test_size ),random_state=1) 
    return X_train,X_dev,X_test, y_train, y_dev, y_test


#train the model of choice with the model parameter

def train_model(x,y,model_params, model_type = "svm"):
    if model_type == "svm":
        clf = svm.SVC
    elif model_type == "tree":
        clf = DecisionTreeClassifier

    model = clf(**model_params)
    #train the model
    model.fit(x,y)
    return model






#find all utils here
def predict_and_eval(model, X_test,y_test):
    predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test,predicted)



def tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination, model_type="svm"):
    best_accuracy_so_far = -1
    best_model_so_far = None
    best_model_path_so_far = ""
    # best_hparams_so_far 
    for itr in list_of_all_param_combination:
    # Model trainging
        cur_model = train_model(X_train, y_train,  itr, model_type="svm")
        
        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)

        # selecting those hparams which give the best perf on dev data set
        if cur_accuracy > best_accuracy_so_far:
            # print("New best accuracy: ", cur_accuracy)
            best_accuracy_so_far = cur_accuracy
            best_hparams_so_far = itr
            best_model_path_so_far ="./{}_".format(model_type) +"_".join(["{}:{}".format(k,v) for k,v in itr.items()]) + ".joblib"
            best_model_so_far = cur_model

    dump(best_model_so_far, best_model_path_so_far) 
    return best_model_so_far,best_accuracy_so_far,best_hparams_so_far



# def tune_hparams_dt_svm(X_train, y_train,params, model_type ):
#     best_accuracy_so_far = -1
#     best_model_so_far = None

#     # Model trainging
#     cur_model = train_model(X_train, y_train,  {}, model_type="decision_tree")
#     cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)

#     return  cur_model,cur_accuracy,



