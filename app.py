import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration


hf_token = st.secrets["HUGGINGFACE_TOKEN"]

model_name = "jaidixit07/streamlit_deploy"
token_name = "jaidixit07/token"
tokenizer = T5Tokenizer.from_pretrained(token_name, token=hf_token)
model = T5ForConditionalGeneration.from_pretrained(model_name, token=hf_token)

# Streamlit interface
st.set_page_config(page_title="Advanced Text Summarization", page_icon="📝", layout="wide")

# Header
st.title("Text Summarization with T5 👨‍💻")
st.markdown("### Utilize the power of the T5 model to generate concise summaries from your input text.")

# Sidebar for input and options
with st.sidebar:
    st.header("Input Options")
    text = st.text_area("Enter text to summarize")
    
    st.markdown("### Summary Options")
    max_length = st.slider("Max Length", min_value=50, max_value=300, value=150, step=10)
    min_length = st.slider("Min Length", min_value=20, max_value=100, value=40, step=5)
    length_penalty = st.slider("Length Penalty", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    num_beams = st.slider("Number of Beams", min_value=1, max_value=10, value=4)

# Main area for results
with st.container():
    if st.button("Summarize"):
        with st.spinner("Generating summary..."):
            inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
            summary_ids = model.generate(
                inputs, 
                max_length=max_length, 
                min_length=min_length, 
                length_penalty=length_penalty, 
                num_beams=num_beams, 
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        st.success("Summary generated successfully!")
        st.write("### Summary:", summary)

# Footer
st.markdown("---")
st.write("Developed by **Jai Dixit**")
st.write("Powered by Hugging Face's T5 model")


# By - Jai Dixit
