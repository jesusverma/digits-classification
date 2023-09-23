from utils import get_hyperparameter_combinations, split_train_dev_test, read_digits


def test_for_hparam_combinations_count():
    # test for checking all possible parameters are generated

    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    c_ranges = [0.1,1,2,5,10]
    h_params={}
    h_params['gamma'] = gamma_ranges
    h_params['C'] = c_ranges
    h_params_combinations = get_hyperparameter_combinations(h_params)

    assert len(h_params_combinations) == len(gamma_ranges) * len(c_ranges)


def test_for_hparam_combinations_values():

    gamma_ranges = [0.001, 0.01]
    c_ranges = [1]
    h_params={}
    h_params['gamma'] = gamma_ranges
    h_params['C'] = c_ranges
    h_params_combinations = get_hyperparameter_combinations(h_params)

    expected_param_combination_1 = {'gamma' : 0.001, 'C':1}
    expected_param_combination_2 = {'gamma' : 0.01, 'C':1}

    assert (expected_param_combination_1 in h_params_combinations) and (expected_param_combination_2 in h_params_combinations)




# def test_data_split():
#     X,y = read_digits()

#     X = X[:100,:,:]
#     y = y[:100]

#     test_size = .1
#     dev_size = .6

#     train_size = 1 - (test_size + dev_size)
#     X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X,y, dev_size=dev_size , test_size=test_size)
    
#     assert(len(X_train) == int(train_size * len(X))) and (len(X_test) == int(test_size * len(X))) and ((len(X_dev) == int(dev_size * len(X))))