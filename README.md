# 🧠 Deep Audio Classifier

This project uses a Convolutional Neural Network (CNN) built with TensorFlow to identify specific audio events from audio recordings. The model is trained on labeled `.wav` audio clips (e.g., a target sound vs. background noise) and can then perform inference on long-format `.mp3` recordings to count the occurrences of that sound.

The core of the project involves converting audio signals into **spectrograms** (visual representations of sound) and training a CNN to recognize the specific visual patterns of the target audio event.



---

## ✨ Features

* **Audio Processing:** Uses `librosa` to load and resample audio files to 16kHz mono.
* **Data Pipeline:** Employs `tf.data` for a highly efficient input pipeline (caching, prefetching, and batching).
* **Spectrogram Generation:** Converts audio waves into spectrograms using `tf.signal.stft`.
* **CNN Model:** A lightweight `tensorflow.keras.Sequential` model for binary classification of images (spectrograms).
* **Inference on Long Audio:** A robust inference workflow that:
    1.  Loads long `.mp3` recordings.
    2.  Splits them into 3-second chunks.
    3.  Preprocesses each chunk into a spectrogram.
    4.  Runs batch predictions on all chunks.
* **Smart Post-processing:** Uses `itertools.groupby` to group consecutive positive predictions, allowing the script to count distinct "audio events" rather than just every 3-second chunk that contains the sound.
* **CSV Reporting:** Saves the final counts of detected events for each recording to a `results.csv` file.

---

## 🛠️ How It Works

1.  **Load Data:** The script loads **positive (target sound)** and **negative (background/other sounds)** `.wav` files from their respective folders into a `tf.data.Dataset`.
2.  **Preprocessing:** A `preprocess` function is mapped across the dataset. It:
    * Loads the audio wave.
    * Pads or truncates it to 3 seconds (48,000 samples).
    * Generates a spectrogram using `tf.signal.stft`.
    * Resizes the spectrogram to `(128, 128)` to be a consistent input for the CNN.
3.  **Training:** The `tf.data` pipeline shuffles, batches, and feeds the spectrograms to the Keras CNN model. The model learns to distinguish between spectrograms containing the target audio event and those that do not.
4.  **Inference:**
    * The script loads each `.mp3` from the inference folder.
    * `tf.keras.utils.timeseries_dataset_from_array` is used to slice the long audio file into 3-second (48,000-sample) windows.
    * Each slice is preprocessed into a 128x128 spectrogram.
    * The trained model predicts on all slices in batches.
5.  **Post-processing & Output:**
    * Raw probability predictions are converted to `0` or `1`.
    * The `groupby` function condenses sequences like `[0, 1, 1, 1, 0, 0, 1]` into `[0, 1, 0, 1]`.
    * The total number of `1`s (distinct audio events) is summed for each file.
    * The final results are written to `results.csv`.

---

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.9+
* `ffmpeg` (for `librosa` to load MP3 files)
    * **macOS (via Homebrew):** `brew install ffmpeg`
    * **Linux (via apt):** `sudo apt update && sudo apt install ffmpeg`

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    tensorflow
    librosa
    numpy
    matplotlib
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Data Structure

This script assumes a specific folder structure, as hardcoded in the file paths.

**Note:** The script uses the folder names `Parsed_Capuchinbird_Clips` and `Parsed_Not_Capuchinbird_Clips` for positive and negative samples, respectively. You can place **any** target audio data in these folders to train your own classifier.
