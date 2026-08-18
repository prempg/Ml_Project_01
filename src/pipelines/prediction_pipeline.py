import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    class CustomData:
      def __init__(
          self,
          hours_studied,
          attendance,
          parental_involvement,
          access_to_resources,
          extracurricular_activities,
          sleep_hours,
          previous_scores,
          motivation_level,
          internet_access,
          tutoring_sessions,
          family_income,
          teacher_quality,
          school_type,
          peer_influence,
          physical_activity,
          learning_disabilities,
          parental_education_level,
          distance_from_home,
          gender
      ):

        self.hours_studied = hours_studied
        self.attendance = attendance
        self.parental_involvement = parental_involvement
        self.access_to_resources = access_to_resources
        self.extracurricular_activities = extracurricular_activities
        self.sleep_hours = sleep_hours
        self.previous_scores = previous_scores
        self.motivation_level = motivation_level
        self.internet_access = internet_access
        self.tutoring_sessions = tutoring_sessions
        self.family_income = family_income
        self.teacher_quality = teacher_quality
        self.school_type = school_type
        self.peer_influence = peer_influence
        self.physical_activity = physical_activity
        self.learning_disabilities = learning_disabilities
        self.parental_education_level = parental_education_level
        self.distance_from_home = distance_from_home
        self.gender = gender
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Hours_Studied": [self.hours_studied],
                "Attendance": [self.attendance],
                "Parental_Involvement": [self.parental_involvement],
                "Access_to_Resources": [self.access_to_resources],
                "Extracurricular_Activities": [
                    self.extracurricular_activities
                ],
                "Sleep_Hours": [self.sleep_hours],
                "Previous_Scores": [self.previous_scores],
                "Motivation_Level": [self.motivation_level],
                "Internet_Access": [self.internet_access],
                "Tutoring_Sessions": [self.tutoring_sessions],
                "Family_Income": [self.family_income],
                "Teacher_Quality": [self.teacher_quality],
                "School_Type": [self.school_type],
                "Peer_Influence": [self.peer_influence],
                "Physical_Activity": [self.physical_activity],
                "Learning_Disabilities": [self.learning_disabilities],
                "Parental_Education_Level": [
                    self.parental_education_level
                ],
                "Distance_from_Home": [self.distance_from_home],
                "Gender": [self.gender],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
