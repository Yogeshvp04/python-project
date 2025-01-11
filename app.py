import json
from flask import Flask, request, render_template
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, Emotion
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Initialize Flask app
app = Flask(__name__)

# IBM Watson NLU setup
apikey = 'your-api-key-here'  # Replace with your Watson NLU API key
url = 'your-url-here'          # Replace with your Watson NLU service URL

# Authenticate Watson NLU
authenticator = IAMAuthenticator(apikey)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlu.set_service_url(url)

# Function to analyze emotions
def analyze_emotions(text):
    try:
        response = nlu.analyze(
            text=text,
            features=Features(emotion=Emotion())
        ).get_result()
        
        emotions = response['emotion']['document']['emotion']
        return emotions
    except Exception as e:
        return {"error": str(e)}

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for emotion detection
@app.route('/detect_emotions', methods=['POST'])
def detect_emotions():
    user_input = request.form['text']
    emotions = analyze_emotions(user_input)
    
    if emotions and "error" not in emotions:
        return render_template('index.html', emotions=emotions)
    else:
        return render_template('index.html', emotions=None, error="Error analyzing emotions. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)
