from flask import Flask, render_template, request
import speech_recognition as sr
from textblob import TextBlob
import csv
import os
from datetime import datetime

app = Flask(__name__)

# 🎤 Speech to Text
def transcribe_audio(file_path):
    r = sr.Recognizer()

    with sr.AudioFile(file_path) as source:
        audio = r.record(source)

    try:
        text = r.recognize_google(audio)
    except:
        text = "Could not understand audio"

    return text


# 🧠 Intent Detection
def detect_intent(text):
    text = text.lower()

    if "refund" in text:
        return "Refund Request"
    elif "problem" in text or "issue" in text:
        return "Complaint"
    elif "price" in text:
        return "Pricing Query"
    else:
        return "General Inquiry"


# 😤 Frustration Detection
def detect_frustration(polarity, text):
    negative_words = ["angry", "bad", "worst", "hate", "issue", "problem"]

    score = 0

    if polarity < -0.3:
        score += 1

    for word in negative_words:
        if word in text.lower():
            score += 1

    return "High" if score >= 2 else "Low"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]

    # Save file
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", "audio.wav")
    file.save(filepath)

    # 🎤 Transcription
    transcript = transcribe_audio(filepath)

    # 😊 Sentiment
    blob = TextBlob(transcript)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # 🎭 Emotion
    if "happy" in transcript.lower():
        emotion = "Happy"
    elif "sad" in transcript.lower():
        emotion = "Sad"
    else:
        emotion = "Neutral"

    # 😤 Frustration
    frustration = detect_frustration(polarity, transcript)

    # 🎯 Intent
    intent = detect_intent(transcript)

    # 💡 Recommendation
    if frustration == "High":
        recommendation = "Escalate to senior support"
    else:
        recommendation = "Handle normally"

    # 📊 SAVE CSV (UTF-8 FIX)
    os.makedirs("reports", exist_ok=True)

    with open("reports/report.csv", "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            datetime.now(),
            transcript,
            sentiment,
            emotion,
            frustration,
            intent
        ])

    return render_template(
        "result.html",
        transcript=transcript,
        sentiment=sentiment,
        emotion=emotion,
        frustration=frustration,
        intent=intent,
        recommendation=recommendation
    )


if __name__ == "__main__":
    app.run(debug=True)