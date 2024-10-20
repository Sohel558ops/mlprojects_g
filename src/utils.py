import os
import sys
import dill

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, y_train, X_test, y_test,models,params):
    try:
        

        report ={}

        for i in range(len(list(models))):
            model =list(models.values())[i]
            para =params[list(models.keys())[i]]


            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            
            
            model.set_params(**gs.best_params_)

            model.fit(X_train,y_train) # Train model
            
            #Make prediction
            y_train_pred = model.predict(X_train)
            y_test_pred= model.predict(X_test)
            
            #Evaluate Train and Test dataset
            #model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
            #model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    