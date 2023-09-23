
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, svm


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
        x,y, test_size=0.5,random_state=random_state
    )
    return X_train, X_test, y_train, y_test



def split_train_dev_test(x,y,dev_size,test_size,random_state=1):
    train_ratio = 1 - (dev_size + test_size)
    X_train, X_test_n_dev, y_train, y_test_n_dev = train_test_split(
        x,y, test_size=1-train_ratio,random_state=random_state
    )
    # test is now (test_size * 100)% of the initial data set
    # dev is now (dev_size * 100)% of the initial data set
    X_dev, X_test, y_dev, y_test = train_test_split(X_test_n_dev, y_test_n_dev, test_size=test_size/(test_size + dev_size)) 
    return X_train,X_dev,X_test, y_train, y_dev, y_test


#train the model of choice with the model parameter

def train_model(x,y,model_params, model_type = "svm"):
    if model_type == "svm":
    #create a support vector classfier
        clf = svm.SVC
    model = clf(**model_params)
    #train the model
    model.fit(x,y)
    return model


#find all utils here
def predict_and_eval(model, X_test,y_test):
    predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test,predicted)



def tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination):
    best_accuracy_so_far = -1
    best_model_so_far = None

    for itr in list_of_all_param_combination:
    # Model trainging
        cur_model = train_model(X_train, y_train,  {'gamma':0.001 , 'C' : itr["c_range"]}, model_type="svm")
        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)

        # selecting those hparams which give the best perf on dev data set
        if cur_accuracy > best_accuracy_so_far:
            # print("New best accuracy: ", cur_accuracy)
            best_accuracy_so_far = cur_accuracy
            optimal_gamma = itr["gamma_range"]
            optimal_c =  itr["c_range"]
            best_model_so_far = cur_model
    return  best_model_so_far,best_accuracy_so_far,optimal_gamma,optimal_c






