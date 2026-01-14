import os
import streamlit as st
import pandas as pd
from predict_email import predict_email

st.set_page_config(page_title="AI Smart Email Classifier", layout="centered")

st.title("📧 AI-Powered Smart Email Classifier")

email_text = st.text_area("Enter Email Content")

if st.button("Classify Email"):
    if email_text.strip() == "":
        st.warning("Please enter email text")
    else:
        category, urgency = predict_email(email_text)

        st.success("Prediction Completed")
        st.write("### Category:", "Spam" if category == 1 else "Not Spam")
        st.write("### Urgency:", urgency.upper())

st.markdown("---")

st.subheader("📊 Sample Email Dataset")
df = pd.read_csv("data/processed/emails_with_urgency.csv").head(20)
st.dataframe(df)
