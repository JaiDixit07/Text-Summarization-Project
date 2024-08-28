import streamlit as st
from transformers import AutoTokenizer, pipeline
from textSummarization.pipeline.prediction import PredictionPipeline  # Assuming your PredictionPipeline is in a file named prediction_pipeline.py

# Initialize the Prediction Pipeline
predictor = PredictionPipeline()

# Streamlit app title and description
st.title("Text Summarization with NLP")
st.write("This application generates a summary for the given text using a pre-trained NLP model.")

# Input text area
input_text = st.text_area("Enter the text you want to summarize:", height=300)

# Summarize button
if st.button("Summarize"):
    if input_text:
        # Generate the summary using the prediction pipeline
        summary = predictor.predict(input_text)
        # Display the original text and summary
        st.subheader("Original Text:")
        st.write(input_text)
        
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.write("Please enter some text to summarize.")

# Footer
st.write("Developed by Jai Dixit")
