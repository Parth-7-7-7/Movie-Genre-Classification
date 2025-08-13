from flask import Flask, request, render_template
import joblib
import re

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model and vectorizer
print("Loading model and vectorizer...")
model = joblib.load('genre_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
print("Model and vectorizer loaded.")

# Text cleaning function (must be the same as used during training)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

@app.route('/', methods=['GET', 'POST'])
def predict_genre():
    prediction = ""
    plot_summary = ""
    # This block runs when the user submits the form
    if request.method == 'POST':
        plot_summary = request.form['plot_summary']
        if plot_summary:
            # 1. Clean the input text
            cleaned_plot = clean_text(plot_summary)
            # 2. Vectorize the text
            vectorized_plot = vectorizer.transform([cleaned_plot])
            # 3. Predict the genre
            prediction = model.predict(vectorized_plot)[0]

    # Render the HTML page, passing the prediction to it
    return render_template('index.html', plot=plot_summary, genre=prediction)

if __name__ == '__main__':
    # Run the app on a local server
    app.run(debug=True)