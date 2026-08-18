from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            hours_studied=float(
                request.form.get("Hours_Studied")
            ),

            attendance=float(
                request.form.get("Attendance")
            ),

            parental_involvement=request.form.get(
                "Parental_Involvement"
            ),

            access_to_resources=request.form.get(
                "Access_to_Resources"
            ),

            extracurricular_activities=request.form.get(
                "Extracurricular_Activities"
            ),

            sleep_hours=float(
                request.form.get("Sleep_Hours")
            ),

            previous_scores=float(
                request.form.get("Previous_Scores")
            ),

            motivation_level=request.form.get(
                "Motivation_Level"
            ),

            internet_access=request.form.get(
                "Internet_Access"
            ),

            tutoring_sessions=float(
                request.form.get("Tutoring_Sessions")
            ),

            family_income=request.form.get(
                "Family_Income"
            ),

            teacher_quality=request.form.get(
                "Teacher_Quality"
            ),

            school_type=request.form.get(
                "School_Type"
            ),

            peer_influence=request.form.get(
                "Peer_Influence"
            ),

            physical_activity=float(
                request.form.get("Physical_Activity")
            ),

            learning_disabilities=request.form.get(
                "Learning_Disabilities"
            ),

            parental_education_level=request.form.get(
                "Parental_Education_Level"
            ),

            distance_from_home=request.form.get(
                "Distance_from_Home"
            ),

            gender=request.form.get("Gender")
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        

