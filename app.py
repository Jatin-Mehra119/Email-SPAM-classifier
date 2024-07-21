from preprocessing import EmailToWordCounterTransformer, WordCounterToVectorTransformer
import joblib
from sklearn.pipeline import Pipeline
import streamlit as st
# Load the pre-trained model
pipeline = joblib.load('spam_ham_model.pkl')

# Function to predict if an email is spam or ham
def predict_spam_ham(email):
    if len(email) == 0:
        return "Please enter an email"
    else:
        prediction = pipeline.predict([email])[0]
        prob = pipeline.predict_proba([email])[0]
        return {
            "prediction": "Spam" if prediction == 1 else "Ham",
            "probability": prob[1] if prediction == 1 else prob[0]
        }

# Streamlit app
st.set_page_config(page_title="Email Spam/Ham Classifier", layout="centered")

st.title("📧 Email Spam/Ham Classifier")
st.write(
    """
    Welcome to the Email Spam/Ham Classifier. Enter the text of your email below to find out if it is considered spam or ham.
    """
)

# Email input area
email = st.text_area("✉️ Enter email:", height=200, placeholder="Type or paste your email content here...")

# Predict button
if st.button("Predict"):
    with st.spinner("Analyzing the email..."):
        result = predict_spam_ham(email)
        if result == "Please enter an email":
            st.warning(result)
        else:
            st.subheader("Prediction Results")
            if result["prediction"] == "Spam":
                st.error(f"**Prediction:** {result['prediction']}")
                st.error(f"**Spam Probability:** {result['probability']:.2f}")
            else:
                st.success(f"**Prediction:** {result['prediction']}")
                st.success(f"**Ham Probability:** {result['probability']:.2f}")

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        color: #333;
    }
    </style>
    <div class="footer">
        Developed by Jatin Mehra. Powered by Streamlit.
        <a href="https://github.com/Jatin-Mehra119" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True
)