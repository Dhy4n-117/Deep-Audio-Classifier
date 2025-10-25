import os
import librosa
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from itertools import groupby

# Define file paths
CAPUCHIN_FILE = os.path.join('data', 'Parsed_Capuchinbird_Clips', 'XC3776-3.wav')
NOT_CAPUCHIN_FILE = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips', 'afternoon-birds-song-in-forest-0.wav')

# Function to load wav file using librosa
def load_wav_16k_mono(filename):
    wav, sample_rate = librosa.load(filename.numpy().decode(), sr=16000, mono=True)
    return wav

# Function wrapper to handle TensorFlow tensors
def load_wav_wrapper(filename):
    wav = tf.py_function(load_wav_16k_mono, [filename], tf.float32)
    wav.set_shape([None])
    return wav

# Load wave files
wave = librosa.load(CAPUCHIN_FILE, sr=16000, mono=True)[0]
nwave = librosa.load(NOT_CAPUCHIN_FILE, sr=16000, mono=True)[0]

# Plot the wave files
plt.plot(wave, label='Capuchinbird')
plt.plot(nwave, label='Not Capuchinbird')
plt.legend()
plt.show()

# Define directories
POS = os.path.join('data', 'Parsed_Capuchinbird_Clips')
NEG = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips')

# Load positive and negative files
pos_files = [os.path.join(POS, f) for f in os.listdir(POS)]
neg_files = [os.path.join(NEG, f) for f in os.listdir(NEG)]

# Create datasets with labels
positives = tf.data.Dataset.from_tensor_slices((pos_files, tf.ones(len(pos_files))))
negatives = tf.data.Dataset.from_tensor_slices((neg_files, tf.zeros(len(neg_files))))
data = positives.concatenate(negatives)

# Preprocessing function
def preprocess(file_path, label):
    wav = load_wav_wrapper(file_path)
    wav = wav[:48000]
    wav = tf.pad(wav, paddings=[[0, 48000 - tf.shape(wav)[0]]], mode='CONSTANT')
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    spectrogram = tf.image.resize(spectrogram, [128, 128])  # Resize to consistent shape
    return spectrogram, label

# Apply preprocessing and prepare datasets
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(8)  # Reduce batch size to reduce memory usage
data = data.prefetch(4)

train = data.take(36)
test = data.skip(36).take(15)

# Define the model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    Conv2D(16, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),  # Reduce number of neurons
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')])
model.summary()

# Train the model
hist = model.fit(train, epochs=4, validation_data=test)

# Plot training history
plt.title('Loss')
plt.plot(hist.history['loss'], 'r', label='Training Loss')
plt.plot(hist.history['val_loss'], 'b', label='Validation Loss')
plt.legend()
plt.show()

plt.title('Precision')
plt.plot(hist.history['precision'], 'r', label='Training Precision')
plt.plot(hist.history['val_precision'], 'b', label='Validation Precision')
plt.legend()
plt.show()

plt.title('Recall')
plt.plot(hist.history['recall'], 'r', label='Training Recall')
plt.plot(hist.history['val_recall'], 'b', label='Validation Recall')
plt.legend()
plt.show()

# Predict on test data
X_test, y_test = test.as_numpy_iterator().next()
yhat = model.predict(X_test)
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]

# Function to load mp3 file and convert to 16kHz mono using librosa
def load_mp3_16k_mono(filename):
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
    return wav

# Load and preprocess mp3 file
mp3 = os.path.join('data', 'Forest Recordings', 'recording_00.mp3')
wav = load_mp3_16k_mono(mp3)

def preprocess_mp3(sample, index):
    sample = tf.reshape(sample, [-1])  # Flatten to 1D tensor
    sample = sample[:48000]
    sample = tf.pad(sample, paddings=[[0, 48000 - tf.shape(sample)[0]]], mode='CONSTANT')
    spectrogram = tf.signal.stft(sample, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    spectrogram = tf.image.resize(spectrogram, [128, 128])  # Resize to consistent shape
    return spectrogram


audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
audio_slices = audio_slices.map(preprocess_mp3)
audio_slices = audio_slices.batch(32)  # Reduce batch size to reduce memory usage

yhat = model.predict(audio_slices)
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]

# Remove consecutive duplicates
yhat = [key for key, group in groupby(yhat)]
calls = tf.math.reduce_sum(yhat).numpy()
print(calls)

# Process all recordings
results = {}
for file in os.listdir(os.path.join('data', 'Forest Recordings')):
    FILEPATH = os.path.join('data', 'Forest Recordings', file)
    wav = load_mp3_16k_mono(FILEPATH)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(32)  # Reduce batch size to reduce memory usage
    yhat = model.predict(audio_slices)
    results[file] = yhat

# Postprocess results
class_preds = {file: [1 if prediction > 0.5 else 0 for prediction in logits] for file, logits in results.items()}
postprocessed = {file: tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy() for file, scores in class_preds.items()}

# Write results to CSV
import csv
with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['recording', 'capuchin_calls'])
    for key, value in postprocessed.items():
        writer.writerow([key, value])

print("Results saved to results.csv")
