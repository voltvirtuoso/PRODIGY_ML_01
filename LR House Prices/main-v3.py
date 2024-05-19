import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the training dataset
train_data = pd.read_csv('train.csv')

# Select relevant features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'SalePrice']
train_data = train_data[features]

# Drop rows with missing values
train_data = train_data.dropna()

# Split features and target variable
target = 'SalePrice'
X_train = train_data.drop(columns=target)
y_train = train_data[target]

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Define the make_prediction function with the updated model
def make_prediction(house_id):
    # Load the test dataset
    test_data = pd.read_csv('test.csv')
    
    # Select features for the given house ID
    house_features = test_data[test_data['Id'] == house_id][['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]
    
    # Make prediction
    prediction = model.predict(house_features)
    
    return house_id, prediction[0]

# Sample submission data
submission_data = {
    "Id": [1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473],
}

# Make predictions for sample submission
predictions = []
for house_id in submission_data['Id']:
    house_id, prediction = make_prediction(house_id)
    predictions.append((house_id, prediction))

# Create DataFrame for submission
submission_df = pd.DataFrame(predictions, columns=['Id', 'SalePrice'])
print(submission_df)
