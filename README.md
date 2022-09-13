# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

first we can take the dataset based on one input value and some mathematical calculaus output value.next define the neaural network model in three layers.first layer have four neaurons and second layer have three neaurons,third layer have two neaurons.the neural network model take inuput and produce actual output using regression.

## Neural Network Model

![image](https://user-images.githubusercontent.com/103049243/189942401-e913281c-2572-46c0-9a3e-59815c9e2733.png)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```python
# Importing Required packages

from google.colab import auth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import gspread
import pandas as pd
from google.auth import default
import pandas as pd

# Authenticate the Google sheet

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('dataset').sheet1

# Construct Data frame using Rows and columns

rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df.head()
df=df.astype({'X':'float'})
df=df.astype({'Y':'float'})
X=df[['X']].values
Y=df[['Y']].values

# Split the testing and training data

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=50)
scaler=MinMaxScaler()
scaler.fit(x_train)
x_t_scaled = scaler.transform(x_train)
x_t_scaled

# Build the Deep learning Model

ai_brain = Sequential([
    Dense(4,activation='relu'),
    Dense(3,activation='relu'),
    Dense(2,activation='relu'),
    Dense(1,activation='relu')
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x=x_t_scaled,y=y_train,epochs=20000)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()

# Evaluate the Model

scal_x_test=scaler.transform(x_test)
ai_brain.evaluate(scal_x_test,y_test)
input=[[105]]
input_scaled=scaler.transform(input)
ai_brain.predict(input_scaled)
```
## Dataset Information
![image](https://user-images.githubusercontent.com/103049243/189944349-340dfee8-84d9-4510-b0f6-64d77e6be7d6.png)
## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/103049243/189944507-9312dd44-2fb6-463a-bbf0-44ff868891e1.png)
### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/103049243/189944455-6c9981c7-efab-486e-bca3-bf7fa074ab81.png)
### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/103049243/189944603-01dab141-fda9-4497-b483-246bc0371d0f.png)
## RESULT
Thus the Neural network for Regression model is Implemented successfully.
