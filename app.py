from flask import Flask, render_template, request, session
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandas.io.json import json_normalize
import csv
import os
# import matplotlib.pyplot as plt

# Sentiment analysis function using VADER
def vader_sentiment_scores(data_frame):
    # Define SentimentIntensityAnalyzer object of VADER.
    SID_obj = SentimentIntensityAnalyzer()

    # calculate polarity scores which gives a sentiment dictionary,
    # Contains pos, neg, neu, and compound scores.
    sentiment_list = []
    for row_num in range(len(data_frame)):
        sentence = data_frame['review'][row_num]

        polarity_dict = SID_obj.polarity_scores(sentence)

        # Calculate overall sentiment by compound score
        if polarity_dict['compound'] >= 0.05:
            sentiment_list.append("Positive")

        elif polarity_dict['compound'] <= - 0.05:
            sentiment_list.append("Negative")

        else:
            sentiment_list.append("Neutral")

    data_frame['Sentiment'] = sentiment_list

    return data_frame


#*** Backend operation
# Read comment csv data
# df = pd.read_csv('data/comment.csv')

# WSGI Application
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name
app = Flask(__name__, template_folder='templates')

app.secret_key = 'You Will Never Guess'

# @app.route('/')
# def welcome():
#     return "Ths is the home page of Flask Application"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        uploaded_file = request.files['uploaded-file']
        df = pd.read_csv(uploaded_file,usecols=['id','language','review'], encoding='unicode_escape')
        
        session['uploaded_csv_file'] = df.to_json()
        return render_template('index2.html')

@app.route('/show_data',methods=("POST", "GET"))
def showData():
    # Get uploaded csv file from session as a json value
    uploaded_json = session.get('uploaded_csv_file', None)
    # Convert json to data frame
    uploaded_json = uploaded_json.replace('true', 'True')
    uploaded_json = uploaded_json.replace('false', 'False')
    
    uploaded_df = pd.DataFrame.from_dict(eval(str(uploaded_json)))
    
    # Convert dataframe to html format
    uploaded_df_html = uploaded_df.to_html()
    return render_template('show_data.html', data=uploaded_df_html)

@app.route('/sentiment', methods=("POST", "GET"))
def SentimentAnalysis():
    # Get uploaded csv file from session as a json value
    uploaded_json = session.get('uploaded_csv_file', None)
    uploaded_json = uploaded_json.replace('true', 'True')
    uploaded_json = uploaded_json.replace('false', 'False')
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(eval(str(uploaded_json)))
    # Apply sentiment function to get sentiment score
    uploaded_df_sentiment = vader_sentiment_scores(uploaded_df)
    uploaded_df_html = uploaded_df_sentiment.to_html()
    return render_template('show_data.html', data=uploaded_df_html)


# @app.route('/graph', methods=("POST", "GET"))
# def graphicalRep():
      
    

#     # Group the results by sentiment
#     sentiment_counts = uploaded_df_sentiment.groupby('sentiment').count()

#     # Create a bar chart of the sentiment counts
#     fig, ax = plt.subplots()
#     sentiment_counts.plot(kind='bar', ax=ax)

#     # Set the chart title and axis labels
#     ax.set_title('Sentiment Analysis Results')
#     ax.set_xlabel('Sentiment')
#     ax.set_ylabel('Count')

#     # Save the chart to a PNG image in memory
#     img = io.BytesIO()
#     plt.savefig(img, format='png')
#     img.seek(0)

#     # Encode the PNG image in base64
#     chart_url = base64.b64encode(img.getvalue()).decode()

#     # Render the HTML template with the chart embedded
#     return render_template('graphd.html', chart_url=chart_url)


if __name__=='__main__':
    app.run(debug = True)