

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

df = pd.read_csv('titanic.csv')

df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

feature = ['Pclass', 'Sex', 'Age', 'FamilySize', 'Fare', 'Embarked', 'Title']
x = pd.get_dummies(df[feature], drop_first=True)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 40)
print(f"🚢 Titanic Random Forest Model Accuracy: {accuracy * 100:.2f}%")
print("=" * 40)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
