
# ESC-50 Audio Classification

A deep learning model trained to classify environmental sounds using the ESC-50 dataset and YamNet embeddings. This project was developed as part of a machine learning hackathon, with a focus on building a working prototype in under 8 hours.

## 🗂️ Overview

The goal is to predict environmental sound classes (e.g., dog barking, rain, siren, chainsaw) from short audio clips. We used:

- Pretrained YamNet to extract embeddings from raw waveforms
- A custom classifier (dense neural network)
- Streamlit for the web app interface

---

## 🎧 Dataset

We used the [ESC-50 dataset](https://github.com/karoldvl/ESC-50) which contains:

- 2,000 labeled audio clips (5 seconds each)
- 50 environmental categories
- Categories grouped into 5 major themes: animals, natural soundscapes, human sounds, interior/domestic sounds, and exterior/urban noises

---

## 🧠 Model Architecture

The model uses transfer learning with YamNet embeddings (1024-dimensional) and a lightweight dense neural net:

- Input: YamNet embedding vector
- Dense (256), ReLU, Dropout
- Dense (128), ReLU, Dropout
- Output: 50-class softmax

We also explored CNNs on log-mel spectrograms and compared results.

---

## 📊 Training Performance

Include your training/validation accuracy/loss plot here.

![Training Accuracy](/path/to/training-accuracy.png)

---

## 📉 Confusion Matrix

Add your final evaluation confusion matrix image here.

![Confusion Matrix](/path/to/confusion-matrix.png)

---

## 🚀 Web App

The project includes a Streamlit web app to demo the classifier:

- Upload a `.wav` file
- Audio is resampled to 16 kHz and fed through YamNet
- Top prediction (and probabilities) shown

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔬 Evaluation Metrics

- Accuracy (on test set): X.XX%
- Top-3 accuracy: X.XX%
- Macro F1 Score: X.XX

Feel free to include the full classification report.

---

## 📁 Project Structure

```
ESC-50-Classifier/
├── app.py                  # Streamlit web app
├── model.h5                # Trained model
├── yamnet.h5               # YamNet model (optional)
├── esc50.csv               # Dataset metadata
├── /audio/                 # Audio files
├── /notebooks/             # Exploration and preprocessing
├── /plots/                 # Training graphs and confusion matrix
└── README.md
```

---

## 📌 Future Work

- Live microphone input for predictions
- Attention-based classifier
- Auto-upload and predict from phone
- Deploy on Hugging Face or Streamlit Cloud

---

## 🙋‍♀️ Credits

- ESC-50 dataset by Karol J. Piczak
- YamNet by Google Research
- Developed by [Your Name or Team Name]
