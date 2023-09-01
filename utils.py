
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, svm



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
    print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
    )
    return predicted


