import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os 

# Load the tokenizer and model from Hugging Face

os.environ['HUGGINGFACE_TOKEN'] = "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"

model_name = "jaidixit07/streamlit_deploy"
token_name="jaidixit07/token"
tokenizer = T5Tokenizer.from_pretrained(token_name, use_auth_token=os.getenv('HUGGINGFACE_TOKEN'))
model = T5ForConditionalGeneration.from_pretrained(model_name, use_auth_token=os.getenv('HUGGINGFACE_TOKEN'), timeout=60)


# Streamlit interface
st.title("Text Summarization with T5")

text = st.text_area("Enter text to summarize")

if st.button("Summarize"):
    # Tokenize and summarize the input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Display the summary
    st.write("Summary:", summary)

# Footer
st.write("Developed by Jai Dixit")
