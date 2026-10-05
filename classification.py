#I used sci-kit because it made the code easier to read but I added hand-coded the methods for the linear regression and metrics at the bottom of the code.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    accuracy_score, precision_score,recall_score,f1_score,roc_auc_score
)

data = pd.read_csv("train.csv")
print(data.head())
print(data.dtypes)

x = data.drop(columns = ["row_id","y","duration"])
x_with_duration = data.drop(columns = "y")
y = data["y"].map({"no":0, "yes":1})

#Ignore 999 in days. 999 does not reflect the true value for the days and each person with that value shouldn't be considered to have the same value for days. 
x["previously_contacted"] = (x["pdays"] != 999).astype(int)
#Replace 999 so i doesn't skew the regression
x["pdays"] = x["pdays"].replace(999, 0)

#Mapping cellular and telephone to binary column because there are only two options
x["cellular"] = x["contact"].map({"cellular": 1, "telephone": 0})
x = x.drop(columns="contact")

#Taking all the other columns that have text values and using one-hot. I chose not to use Sci Kit for one hot but could have.
textColumns = x.select_dtypes(include = ["object"]).columns
x = pd.get_dummies(x,columns = textColumns, dtype =int)

x_train, x_test, y_train, y_test= train_test_split(
    x,y,test_size=0.2,random_state=50, stratify = y
)

model = LogisticRegression(max_iter = 1000)
model.fit(x_train,y_train)

probability_yes = model.predict_proba(x_test)[:,1]


predictions = []

for p in probability_yes:
    if p >= 0.45:
        predictions.append(1)
    else:
        predictions.append(0)
print("Accuracy", accuracy_score(y_test,predictions))
print("Precision", precision_score(y_test,predictions,zero_division=0))
print("Recall", recall_score(y_test, predictions,zero_division=0))
print("F1", f1_score(y_test, predictions,zero_division=0))
print("ROC_AUC", roc_auc_score(y_test,probability_yes))


#here is how it would look without using the sci kit model: 
def sigmoid(z):   #defining function that takes z
    return 1 / (1 + np.exp(-z)) #sigmoid formula, computes e to the power of -z for each value

def fit_logistic(X, y, lr=0.1, epochs=1000):  #defining the training function. epochs is how many times to repeat the update
    X = np.asarray(X, dtype=float)         #convert inputs into numpy arrays
    y = np.asarray(y, dtype=float)
    w = np.zeros(X.shape[1])    #x.shape is the number of features 
    b = 0.0        #bias starts at 0
    for _ in range(epochs):     #repeating the epoch update
        p = sigmoid(X @ w + b)     #turning the value into probability between 0 and 1. 
        w -= lr * X.T @ (p - y) / len(y)
        b -= lr * (p - y).mean()
    return w, b      #returning the learned weights and biases

def predict_probability(X, w, b):     #takes new data set with calculated weights and biases from training
    return sigmoid(np.asarray(X, dtype=float) @ w + b)




