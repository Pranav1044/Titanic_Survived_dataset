import pandas as pd
df = pd.read_csv("Exp2/Titanic_dataset.csv")

from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
df["Cabin"] = Le.fit_transform(df["Cabin"])
df["Embarked"] = Le.fit_transform(df["Embarked"])
# print(Le.classes_)
df["Gender"] = Le.fit_transform(df["Gender"])

X = df.drop(["PassengerId","Survived","Name","Ticket","SibSp","Parch"],axis = 1)
Y = df["Survived"]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=66)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
ac = accuracy_score(Y_test,Y_pred)
print(ac)