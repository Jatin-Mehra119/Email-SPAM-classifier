import joblib
from sklearn.pipeline import Pipeline
from preprocessing import EmailToWordCounterTransformer, WordCounterToVectorTransformer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score

# Load the data
print("loading data.....")
df = pd.read_csv('spam_ham_dataset.csv')
print("data loaded")

# Split the data
print("splitting data.....")
X = df['text'].copy()
y = df['label_num'].copy()

train_X, test_X, train_y, test_y = train_test_split(X, y, stratify=df['label_num'], test_size=0.2, random_state=42)
print(f"data splited \n Training instances : {len(train_X)} \n Testing instances : {len(test_X)}")

# Define transformers
email_transformer = EmailToWordCounterTransformer()
word_vector_transformer = WordCounterToVectorTransformer()

# Create the pipeline
print("training the model.....")
model = Pipeline([
    ('email_transformer', email_transformer),
    ('word_vector_transformer', word_vector_transformer),
    ('model', LogisticRegression(max_iter=1000))
])

model.fit(train_X, train_y)
print("model trained")

print("Starting cross validation.....")
# Cross val Scores
print(f"Cross Validation Scores : {cross_val_score(model, train_X, train_y, cv=3, scoring='accuracy').mean()}")

# Evaluate the model on test data
print("Evaluating the model.....")

test_predictions = model.predict(test_X)
print(f"Accuracy : {accuracy_score(test_y, test_predictions)}")
print(f"Precision : {precision_score(test_y, test_predictions)}")
print(f"Recall : {recall_score(test_y, test_predictions)}")
print(f"F1 Score : {f1_score(test_y, test_predictions)}")
print(f"Classification Report : \n {classification_report(test_y, test_predictions)}")

print("Saving the model.....")

joblib.dump(model, 'spam_ham_model.pkl')