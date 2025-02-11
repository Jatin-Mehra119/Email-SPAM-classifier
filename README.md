##  Email Spam/Ham Classifier App
In this repository, I'm developing a Web App using Streamlit framework and train a machine learning algorithm to achieve the function of the app to predict whether the entered text/E-mail is Spam or not.

## [APP](https://jatin-mehra119-email-spam-classifier-app-dhitia.streamlit.app/)

### For Training the model
- **Data** - I'm using a dataset from Kaggle, which contains labels and email in tabular form. The dataset doesn't contain any errors or invalid data.
- **Visualizing Data**- Using Seaborn, matplotlib.
- **Preprocessing** - 
Using nltk, re, urlextract, counter for text preprocessing
Using custom transformers to convert text to lowercase, replacing URLs with a placeholder "URL", similarly for numbers, removing punctuation, splits text into words and counts occurrences, applying stemming to words, converts word count dictionaries into a sparse matrix for use in machine learning models.
- **Model Training**-
Using a Logistic Regression Model with no hyperparameters tunning(model is already performing good)
- **Model Evaluation**-
Accuracy: 0.97
Precision: 0.94
Recall: 0.96
F1 Score : 0.95
- **Saving Model**-
Using joblib

### For App
- Using streamlit with simple easy to use interactive UI.
- Makes prediction using pre-trained model using joblib.

### Use Cases
- Job Seekers: Candidates can use the app to check if their job application emails are likely to be marked as spam, ensuring better delivery rates to potential employers.
- HR Professionals: HR departments can screen incoming job application emails to filter out spam and prioritize genuine applications.
- Email Marketers: Marketing professionals can test their email campaigns to minimize the risk of their emails being flagged as spam, thereby improving open rates.
- Small Businesses: Entrepreneurs can ensure that their business communications are not mistaken for spam, fostering better client relationships.
- Students and Researchers: Academics can use the app to understand spam detection and apply similar techniques to their own projects or research.
