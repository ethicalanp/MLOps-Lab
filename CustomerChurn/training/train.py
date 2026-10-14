import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import joblib

#load data 
df=pd.read_csv("data/Telco-Customer-Churn.csv")

#data inspection
print("shape",df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\n First 5 rows:")
print(df.head())


print("\n Dataset information:")
print(df.info())

print("missing values :")
print(df.isnull().sum())

print("Duplicate rows: ")
print(df.duplicated().sum())

#target distribution
print("\n Churn distribution: ")
print(df["Churn"].value_counts())

#Cleaning
df["TotalCharges"]==pd.to_numeric(df["TotalCharges"],errors="coerce")

df=df.drop(columns=["customerID"])

df['Churn']=df['Churn'].map({"No":0,"Yes":1})

print("\n" + "=" * 60)
print("Missing Values : ","=" * 60)
print(df.isnull().sum())

#target distribution
print("\n" + "=" * 60)
print("Target Distribution" ,"=" * 60)
print("Target Percentage : ")
print(df["Churn"].value_counts(normalize=True)*100)

#Seperate Features and Target 
X = df.drop(columns=["Churn"])
y = df["Churn"]

print("\n" + "=" * 60)
print("Features")
print(X.columns.tolist())


#Numerical and Categorical Features 
numerical_features=X.select_dtypes(include=["int64","float64"]).columns.tolist()

categorical_features=X.select_dtypes(include=["object"]).columns.tolist()

print("Numerical Features :" ,numerical_features)
print("Categorical Features : ",categorical_features)

#Train Test Split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

print("\n" + "=" * 60)
print("Train test split" )
print('=' * 60)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

#Numerical Preprocessing 
numerical_pipeline=Pipeline(
    steps=[("imputer",SimpleImputer(strategy="median")),
           ("scaler",StandardScaler())]
)

#Categorical Preprocessing 
categorical_pipeline=Pipeline(
    steps=[("imputer",SimpleImputer(strategy="most_frequent")),
           ("encoder",OneHotEncoder(handle_unknown="ignore"))]
)
#Combine Preprocessing
preprocessor=ColumnTransformer(transformers=[("numerical",numerical_pipeline,numerical_features),
                                             ("categorical",categorical_pipeline,categorical_features)])

#Complete ML Pipeline
pipeline=Pipeline(
    steps=[("preprocessor",preprocessor),
           ("model",LogisticRegression(max_iter=1000))]
)

#Train model
print("\n" + "=" * 60)
print("Training Model")
print("=" * 60)

pipeline.fit(X_train,y_train)

print("Training Complete")

#Predictions
y_pred=pipeline.predict(X_test)

#Evaluate
accuracy=accuracy_score(y_test,y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy :{accuracy :.4f}")

print("\n Classification Report")
print(classification_report(y_test,y_pred))

print("\n Confusion Matrix")
print(confusion_matrix(y_test,y_pred))

#Saving Complete Pipline 
MODEL_PATH="model/churn_pipeline.pkl"

joblib.dump(pipeline,MODEL_PATH)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"Saved to : {MODEL_PATH}")
