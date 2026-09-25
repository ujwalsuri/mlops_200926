# for data manipulation
import pandas as pd
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

df = pd.read_csv("mlops/data/tourism.csv")
print("Dataset loaded successfully.")

#drop the CustomerID and unnamed columns from data which are unnecessary
data = df.drop(columns=['CustomerID'])
data.drop("Unnamed: 0", axis=1, inplace=True)
print("No of columns after dropping unnecessary columns: ")
len(data.columns)

#Data corrections:- replace 'Fe Male' with 'Female' and 'Unmarried' with 'Single'
data['Gender'] = data['Gender'].replace({'Fe Male': 'Female'})
data['MaritalStatus'] = data['MaritalStatus'].replace({'Unmarried': 'Single'})

print("No of Duplicate/Identical rows in the data:-")
data.duplicated().sum()

#drop duplicate rows
data = data.drop_duplicates()

print("No of rows after dropping duplicates: ")
len(data)

#drop Designation column as it is perfectly correlated with ProductPitched in current data
data = data.drop(columns=['Designation'])

# Define the target variable for the classification task
target = "ProdTaken"

# List of numerical features in the dataset
numeric_features = ["Age", "NumberOfTrips", "NumberOfPersonVisiting", "NumberOfChildrenVisiting", "DurationOfPitch", "MonthlyIncome", "NumberOfFollowups"]

# List of categorical features in the dataset
categorical_features = ["TypeofContact", "CityTier", "Occupation", "Gender", "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "Passport", "PitchSatisfactionScore", "OwnCar"]

# Define predictor matrix (X) using selected numeric and categorical features
X = df[numeric_features + categorical_features]

# Define target variable
y = df[target]

# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42,   # Ensures reproducibility by setting a fixed random seed
    stratify=y,        # keeps the (imbalanced) ratio consistent across splits
)

#Save split data to individual csvs
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
