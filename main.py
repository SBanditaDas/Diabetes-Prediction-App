import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Loading the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',
           'DiabetesPedigreeFunction','Age','Outcome']
df = pd.read_csv(url, names=columns)

# Replacing 0s with NaN and fill with mean
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = \
    df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

# Preprocessing
X = df.drop('Outcome', axis=1)
y = df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Saving the model
joblib.dump(model, 'diabetes_model.pkl')

# Printing accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))