from utils import get_hyperparameter_combinations, split_train_dev_test, read_digits, preprocess_data, tune_hparams
import os

def test_for_hparam_combinations_count():
    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    c_ranges = [0.1,1,2,5,10]
    h_params={}
    h_params['gamma'] = gamma_ranges
    h_params['C'] = c_ranges
    h_params_combinations = get_hyperparameter_combinations(h_params)

    assert len(h_params_combinations) == len(gamma_ranges) * len(c_ranges)



def get_dummy_data():
    X, y = read_digits()
    
    X_train = X[:100,:,:]
    y_train = y[:100]
    X_dev = X[:50,:,:]
    y_dev = y[:50]

    X_dev = preprocess_data(X_dev)
    X_train = preprocess_data(X_train)

    return X_train, y_train, X_dev, y_dev

def get_dummy_hyper_params():
    gamma_list = [0.001, 0.01]
    C_list = [1]
    h_params={}
    h_params['gamma'] = gamma_list
    h_params['C'] = C_list
    h_params_combinations = get_hyperparameter_combinations(h_params)
    return h_params_combinations

def test_for_hparam_combinations_values():    
    h_p_combinations = get_dummy_hyper_params()
    
    expected_param_1 = {'gamma': 0.001, 'C': 1}
    expected_param_2 = {'gamma': 0.01, 'C': 1}

    assert (expected_param_1 in h_p_combinations) and (expected_param_2 in h_p_combinations)


def test_model_saving():
    X_train, y_train, X_dev, y_dev = get_dummy_data()
    h_params_combinations = get_dummy_hyper_params()

    _, best_model_path, _ = tune_hparams(X_train, y_train, X_dev, 
        y_dev, h_params_combinations)   

    assert os.path.exists(best_model_path)


def test_data_split():
    X,y = read_digits()

    X = X[:100,:,:]
    y = y[:100]

    test_size = .1
    dev_size = .6


    X_train,X_dev,X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, dev_size=dev_size,  test_size=test_size,)

    assert (len(X_train) == 30) 
    assert (len(X_test) == 10)
    assert ((len(X_dev) == 60))