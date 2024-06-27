# Big-Sales-Prediction
Predict the sales of items in a retail dataset using linear regression.


# Big Sales Prediction

## Objective
Predict the sales of items in a retail dataset based on various features such as item weight, visibility, MRP, and the outlet establishment year.

## Data Source
The dataset used for this project can be found at the following link: [Big Sales Data](https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Big%20Sales%20Data.csv)

## Project Steps

1. **Import Libraries**
2. **Import Data**
3. **Describe Data**
4. **Data Visualization**
5. **Data Preprocessing**
6. **Define Target Variable (y) and Feature Variables (X)**
7. **Train Test Split**
8. **Modeling**
9. **Model Evaluation**
10. **Prediction**

## Code
The following is the complete code used in this project:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Big%20Sales%20Data.csv')

# Display the first few rows of the dataset
df.head()

# Display dataset info and statistical summary
df.info()
df.describe()

# Visualize data relationships
sns.pairplot(df)
plt.show()

# Fill missing values for 'Item_Weight'
df['Item_Weight'].fillna(df.groupby('Item_Type')['Item_Weight'].transform('mean'), inplace=True)

# Standardize 'Item_Fat_Content' values
df.replace({'Item_Fat_Content': {'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'}}, inplace=True)
df.replace({'Item_Fat_Content': {'Low Fat': 0, 'Regular': 1}}, inplace=True)

# Simplify 'Item_Type' values
df.replace({'Item_Type': {'Fruits and Vegetables': 0, 'Snack Foods': 0, 'Household': 1, 'Frozen Foods': 1}}, inplace=True)

# Verify preprocessing
df.info()

# Define features and target variable
X = df[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']]
y = df['Item_Outlet_Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Make a sample prediction
sample_input = np.array([[12.3, 0.111448, 33.4874, 1997]])
sample_prediction = model.predict(sample_input)
print(f'Sample Prediction: {sample_prediction}')
