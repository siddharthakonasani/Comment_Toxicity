# Comment Toxicity Detection

This project is a deep learning-based web application for classifying the toxicity of online comments in real time. Built with Streamlit, it leverages natural language processing and an LSTM-based neural network trained to detect various categories of toxic content.

**Live App:**  

https://sid-comment-toxicity.streamlit.app/
---

## Features

- Input any comment and instantly get predictions for:
  - Toxic
  - Severe Toxic
  - Obscene
  - Threat
  - Insult
  - Identity Hate
- Modern, easy-to-use web interface powered by [Streamlit](https://streamlit.io/)
- Model was trained on publicly available toxicity data and uses deep learning and NLP preprocessing

---

## Getting Started

### 1. Clone this Repository

```
git clone https://github.com/siddharthakonasani/Comment_Toxicity.git
cd Comment_Toxicity
```

### 2. Install Requirements

It's recommended to use a fresh virtual environment or conda environment.

```
pip install -r requirements.txt
```


### 3. Run the Streamlit App Locally
```
streamlit run app.py
```


The app will open in your browser, or you can visit the local link printed in your terminal.

---

## Project Structure

Comment_Toxicity/
│
|
├── app.py # Streamlit app code
|
├── best_lstm_model.h5 # Trained model (for inference)
|
├── tokenizer.pickle # Tokenizer (for consistent text preprocessing)
|
├── requirements.txt # Python dependencies
|
├── README.md # This file



---

## Model Details

- The neural network uses a Bidirectional LSTM architecture for text classification.
- Text is preprocessed (cleaned and tokenized) consistent with the training procedure.
- Predictions are shown as the probability for each toxicity label.

---

## Dataset

*Due to size constraints, training data is not included in this repo. To retrain the model, see instructions (if provided) and use your local data.*

---

## Deployment

To deploy on Streamlit Cloud:

1. Fork or push this repository to your GitHub.
2. Sign in to [Streamlit Cloud](https://streamlit.io/cloud).
3. Connect your GitHub and select this repo.
4. Set `app.py` as the entrypoint.  
5. Add the app link above after it goes live!

---

## Credits

- Developed by Siddhartha Ram Konasani. 
- Mail: **siddharthakonasani.77@gmail.com**
- Model and workflow inspired by public toxicity classification challenges.


