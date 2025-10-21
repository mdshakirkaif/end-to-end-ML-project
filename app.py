from flask import Flask,render_template,request
from src.pipeline.predict_pipeline import customData,PredictPipeline

application=Flask(__name__)
app=application

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')
    
@app.route('/predict_datapoint',methods=['GET','POST'])
def predict_datapoints():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=customData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        
        pred_df=data.get_data_dataframe()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        
        return render_template('home.html',results=results[0])
    
if __name__=='__main__':
    app.run(host='0.0.0.0')