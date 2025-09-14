import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create a small dataset
data = {
    "message": [
        "Congratulations! You have won a lottery worth $1,000,000. Claim now!",
        "Hi John, are we still meeting tomorrow at 5pm?",
        "Free entry in a weekly competition to win an iPhone. Click here!",
        "Hey, can you send me the notes from today’s class?",
        "Urgent! Your account has been compromised. Reset your password immediately.",
        "Hi Mom, just reached home safely. Talk to you later.",
        "Win a free vacation to Paris. Text WIN to 12345 now!",
        "Are you available for a quick call regarding the project?",
        "You’ve been selected for a free credit card. Apply today!",
        "Don’t forget our dinner reservation tonight at 8."
    ],
    "label": [
        1, # spam
        0, # ham
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0
    ]
}

df = pd.DataFrame(data)

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.3, random_state=42
)

# Step 3: Feature extraction (Bag of Words)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Step 4: Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Test with new emails
sample_messages = [
    "Claim your free cash prize now!!!",
    "Hey, want to grab lunch today?"
]
sample_features = vectorizer.transform(sample_messages)
predictions = model.predict(sample_features)

for msg, pred in zip(sample_messages, predictions):
    print(f"\nMessage: {msg}\nPrediction: {'SPAM' if pred==1 else 'HAM'}")
