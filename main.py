import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,classification_report,root_mean_squared_error
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from lightgbm import LGBMRegressor


MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def Builded_pipeline(num_attrib,cat_attrib):
    num_pipeline = Pipeline(
    [
        ("impute",SimpleImputer(strategy="median") ),
        ("scaler",StandardScaler())
    ]
    )
    cat_pipeline = Pipeline([
        ("encode",OneHotEncoder(handle_unknown="ignore"))
    ])

    fullpipeline = ColumnTransformer(
        [
            ("num",num_pipeline,num_attrib),
            ("cat",cat_pipeline,cat_attrib)
        ]
    )
    return fullpipeline
if not os.path.exists(MODEL_FILE):
    diabetes_data = pd.read_csv('diabetes_dataset.csv')
    diabetes_data = diabetes_data.drop(["race:AfricanAmerican",	"race:Asian",	"race:Caucasian","race:Hispanic",	"race:Other","location","year"
    ],axis=1)
    diabetes_data["bmi_cat"] = pd.cut(diabetes_data['bmi'],bins=[10.0,15.5,25.0,35.5,45.0,np.inf],labels=[1,2,3,4,5])
    shuffler = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_set,test_set in shuffler.split(diabetes_data,diabetes_data["bmi_cat"]):
        strata_train = diabetes_data.loc[train_set].drop("bmi_cat",axis=1)
        strata_test = diabetes_data.loc[test_set].drop("bmi_cat",axis=1)
    train_data = strata_train.copy()
    train_feature = train_data.drop("diabetes",axis=1)
    train_label = train_data['diabetes']
    test_feature = strata_test.drop("diabetes",axis=1)
    test_label = strata_test["diabetes"]

    num_attrib = train_feature.drop(["gender","smoking_history"],axis=1).columns.tolist()
    cat_attrib = ["gender","smoking_history"]
    new_pipeline = Builded_pipeline(num_attrib,cat_attrib)
    diabetes_data_transformed = new_pipeline.fit_transform(train_feature)
    model =  LGBMRegressor(n_estimators=100,min_samples_split=10,min_samples_leaf=2,max_depth=5,random_state=42)
    model.fit(diabetes_data_transformed,train_label)
    joblib.dump(model,MODEL_FILE)
    joblib.dump(new_pipeline,PIPELINE_FILE)
    print("MODEL HAS BEEN TRAINED")
else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    input_data = pd.read_csv('input.csv')
    data_transformed = pipeline.transform(input_data)
    predicted_output = model.predict(data_transformed)
    input_data['diabetes'] = predicted_output
    input_data.to_csv("prediction.csv",index=False)
    print("CONGRATULATIONS!!!FINALLY INFERENCE COMPLETED,YOU CAN VIEW THE PREDICTION IN PREDICTION.CSV")
    
