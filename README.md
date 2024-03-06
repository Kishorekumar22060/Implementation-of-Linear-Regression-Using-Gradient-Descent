# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for Gradient Design.
2.Upload the dataset and check any null value using .isnull() function.
3.Declare the default values for linear regression.
4.Calculate the loss usinng Mean Square Error.
5.Predict the value of y.
6.Plot the graph respect to hours and scores using scatter plot function.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Kishore Kumar U
RegisterNumber:  212222233003
*/
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):

    # Add a column of ones to X for the intercept term
    X = np.c_[np.ones(len(X1)), X1]

    # Initialize theta with zeros
    theta = np.zeros(X.shape[1]).reshape(-1, 1)

    # Perform gradient descent
    for i in range(num_iters):

        # Calculate predictions
        predictions = np.dot(X, theta).reshape(-1, 1)

        # Calculate errors
        errors = predictions - y.reshape(-1, 1)

        # Update theta using gradient descent
        theta = theta - learning_rate * (1 / len(X1)) * np.dot(X.T,errors)

    return theta


data=pd.read_csv('/content/50_Startups.csv',header=None)
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)


scaler=StandardScaler()

y=(data.iloc [1:, -1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform (X1)
Y1_Scaled=scaler.fit_transform(y)

theta=linear_regression(X1_Scaled,Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)


prediction=np.dot(np.append(1,new_Scaled),theta) 
prediction=prediction.reshape(-1,1)


pre=scaler.inverse_transform(prediction) 
print(f"Predicted value: {pre}")

```

## Output:
![image](https://github.com/AkilaMohan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/141472136/fae53a23-cdd7-416e-93c2-b604a2f49f98)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
