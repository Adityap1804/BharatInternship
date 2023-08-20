import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


train_data=pd.read_csv("/Users/adityapnv/Downloads/train.csv")
print(train_data)
train_data.isnull().sum()

#countplot to check how many Survived
sns.countplot('Survived',data=train_data)
plt.show()

#lets drop cabin column because it has many null values.
train_data.drop('Cabin',axis=1,inplace=True)
#Here we are droping all Nan values
train_data.dropna(inplace=True)
#Let's check it in heatmap 
sns.heatmap(train_data.isnull(),yticklabels=False)
plt.show()


#Data Splitting
X = train_data[["Pclass", "Sex", "Age", "Fare"]]  # Features
y = train_data["Survived"]  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Selection and Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Step 6: Feature Importance
feature_importance = model.feature_importances_
print("Feature Importance:", feature_importance)