# Text Summarization with T5 👨‍💻

Welcome to the Text Summarization project using the T5 model! This Streamlit app allows you to generate concise summaries from input text using the powerful T5 model from Hugging Face. 

## 🚀 Features

- **Advanced Summarization**: Generate summaries with customizable options such as maximum length, minimum length, length penalty, and number of beams.
- **Interactive UI**: The app includes an intuitive and interactive interface powered by Streamlit.
- **Secure API Integration**: Access the Hugging Face API securely using Streamlit Secrets.

## 🛠️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/text-summarization-t5.git
cd text-summarization-t5


Install Dependencies
Ensure you have Python 3.8 or later installed. Then, install the required packages:

pip install -r requirements.txt

Set Up Streamlit Secrets
To securely access the Hugging Face API token, add the token to Streamlit Secrets:

Go to the Streamlit app's dashboard.
In the app settings, find the "Secrets" section.
Add your API token in the TOML format:

HUGGINGFACE_TOKEN = "your_huggingface_api_token"


🖥️ Usage
To run the Streamlit app locally:

bash
Copy code
streamlit run app.py
This will start the app on http://localhost:8501/ where you can interact with the text summarization tool.

🔧 Customization
Summary Options
Max Length: Set the maximum length of the generated summary.
Min Length: Set the minimum length of the generated summary.
Length Penalty: Adjust the penalty for longer sentences (values >1.0 will make the model prefer shorter sequences).
Number of Beams: Set the number of beams for beam search (higher values result in better quality but are slower).
Example Usage
python
Copy code
inputs = tokenizer.encode("summarize: Your input text here", return_tensors="pt", max_length=512, truncation=True)
summary_ids = model.generate(
    inputs, 
    max_length=150, 
    min_length=40, 
    length_penalty=2.0, 
    num_beams=4, 
    early_stopping=True
)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

🙌 Acknowledgements
Hugging Face for providing the T5 model.
Streamlit for the amazing framework that powers this app.