import nltk
nltk.download('stopwords')
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import seaborn as sns
import pickle

df = pd.read_csv('/content/spam.csv', encoding='latin-1')
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df.columns = ["label", "message"]

def preprocess_content(text):
    stemmer = PorterStemmer()
    nopunc = ''.join([char for char in text if char not in string.punctuation])
    words = word_tokenize(nopunc.lower())
    nostop = [stemmer.stem(word) for word in words if word not in stopwords.words('english') and word.isalpha()]
    return ' '.join(nostop)

# Apply preprocessing
df['cleaned_text'] = df['message'].apply(preprocess_content)

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['cleaned_text'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Evaluate model
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
rf_report = classification_report(y_test, rf_preds)

print("Random Forest Model:")
print(f'Accuracy: {rf_accuracy:.4f}')
print(f'Classification Report:\n{rf_report}\n')
