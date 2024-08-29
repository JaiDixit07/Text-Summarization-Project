# Text Summarization with T5 Model 📚✂️

This repository contains a Streamlit application that leverages the Hugging Face T5 model to generate concise summaries from input text. The app is designed to be user-friendly and highly customizable, allowing users to adjust the summarization parameters to their liking.

## 🚀 Features

- **Advanced Summarization**: Generate high-quality summaries using the T5 model.
- **Customizable Parameters**: Adjust max length, min length, length penalty, and the number of beams for beam search to fine-tune your summary.
- **Secure API Access**: The app integrates securely with Hugging Face's API using Streamlit Secrets.

## 🛠️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/text-summarization-t5.git
cd text-summarization-t5
```


### Install Dependencies

Make sure you have Python 3.8 or later installed. Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Configure Streamlit Secrets
To use the Hugging Face API securely, add your API token to Streamlit Secrets:

1. Access your Streamlit app's dashboard.
2. Navigate to the "Secrets" section in app settings.
3. Add your API token in the following TOML format:

```c
HUGGINGFACE_TOKEN = "your_huggingface_api_token"
```

## 🖥️ Usage
To run the Streamlit app locally, execute:

```bash 
streamlit run app.py
```

The app will be accessible at http://localhost:8501/. Here, you can input text and adjust summarization settings to generate a summary.

# 🔧 Customization Options

- **Max Length**: Set the maximum length of the summary.
- **Min Length**: Define the minimum length for the generated summary.
- **Length Penalty**: Penalize longer sentences (higher values prefer shorter summaries).
- **Number of Beams**: The number of beams for beam search, balancing speed and quality of the summary.

## Example Code
Here’s a snippet of how the summarization process is handled within the app:

```python
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
```

## 🤝 Contributing
Contributions are welcome! If you encounter any issues or have feature requests, please open an issue on the [issues page](https://github.com/your-username/text-summarization-t5/issues).

## 📝 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 🙌 Acknowledgements
- **Hugging Face** for the T5 model and transformers library.
- **Streamlit** for the framework that powers this app.

-------------------------------------------------------------------------------------------------

Developed by Jai Dixit.