import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import gradio as gr

model = tf.keras.models.load_model("audio-classifier.keras")

esc50_labels = [
    'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects',
    'sheep', 'crow', 'rain', 'sea_waves', 'crackling_fire', 'crickets',
    'chirping_birds', 'water_drops', 'wind', 'pouring_water', 'toilet_flush',
    'thunderstorm', 'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing',
    'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping',
    'door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks',
    'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick',
    'glass_breaking', 'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine',
    'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw'
]


def extract_yamnet_embedding(waveform):
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return tf.reduce_mean(embeddings, axis=0).numpy()


def predict(uploaded_file):
    waveform, sr = sf.read(uploaded_file)
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    embedding = extract_yamnet_embedding(waveform)
    preds = model.predict(np.expand_dims(embedding, axis=0))[0]
    top_idx = np.argmax(preds)
    top_label = esc50_labels[top_idx]
    return {top_label: preds[top_idx]}


interface = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload or Record Audio"),
    outputs=gr.Label(num_top_classes=3),
    title="Sound Classifier",
    description="Upload or record a short sound to classify."
)

interface.launch()
