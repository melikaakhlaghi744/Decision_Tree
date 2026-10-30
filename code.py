import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,classification_report
df=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
df=df.dropna()
df=df.drop(columns=["customerID"])
features=["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"]
X=df[features].copy()
y=df["Churn"]
X["SeniorCitizen"]=X["SeniorCitizen"].map({0:"No",1:"Yes"})
X["tenure"]=pd.cut(X["tenure"],bins=[-1,12,36,72],labels=["Short","Medium","Long"])
X["MonthlyCharges"]=pd.cut(X["MonthlyCharges"],bins=[0,35,70,120],labels=["Low","Medium","High"],include_lowest=True)
X["TotalCharges"]=pd.cut(X["TotalCharges"],bins=[-1,1000,3000,np.inf],labels=["Low","Medium","High"])
X=pd.get_dummies(X)
y=y.map({"No":0,"Yes":1})
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
model=DecisionTreeClassifier(criterion="gini",max_depth=5,random_state=42,class_weight="balanced")
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred,target_names=["No Churn","Churn"]))
plt.figure(figsize=(25,12))
plot_tree(model,feature_names=X.columns,class_names=["No Churn","Churn"],filled=True,rounded=True,fontsize=7)
plt.title("Decision Tree - Customer Churn")
plt.show()
