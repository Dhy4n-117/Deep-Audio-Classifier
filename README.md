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
    git clone (https://github.com/Dhy4n-117/Deep-Audio-Classifier.git)
    cd Deep-Audio-Classifier
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

Got it. Here is an updated `Datasets & Folder Structure` section for your README. This new section is more detailed and clearly explains what each dataset is for.

You should **replace** the *existing* "3. Data Structure" section in your README with this one.

-----

### 3\. Datasets & Folder Structure

This project requires three sets of data, which you must provide in the following folder structure:

```
Deep-Audio-Classifier/
│
├── data/
│   │
│   ├── Parsed_Capuchinbird_Clips/    # 1. POSITIVE Training Data
│   │   ├── clip1.wav
│   │   └── ...
│   │
│   ├── Parsed_Not_Capuchinbird_Clips/  # 2. NEGATIVE Training Data
│   │   ├── not_clip1.wav
│   │   └── ...
│   │
│   └── Forest Recordings/              # 3. INFERENCE Data
│       ├── recording_00.mp3
│       └── ...
│
├── classifier.py   # The main Python script
└── ...
```

#### 1\. Positive Training Data (`Parsed_Capuchinbird_Clips`)

  * **Purpose:** To teach the model what your target sound *is*.
  * **Format:** Short `.wav` files (ideally 2-5 seconds) that clearly contain the audio event you want to detect.
  * **Label:** The script automatically assigns these a label of `1`.

#### 2\. Negative Training Data (`Parsed_Not_Capuchinbird_Clips`)

  * **Purpose:** To teach the model what your target sound *is not*. This is just as important\!
  * **Format:** Short `.wav` files containing background noise, other similar-sounding (but incorrect) events, or silence.
  * **Label:** The script automatically assigns these a label of `0`.

#### 3\. Inference Data (`Forest Recordings`)

  * **Purpose:** This is the data you want to analyze *after* the model is trained.
  * **Format:** Long-format `.mp3` or `.wav` files that the script will scan for the target sound.
  * **Output:** The script will analyze these files and generate a `results.csv` listing the detected event counts for each file.

**Note:** The folder names `Parsed_Capuchinbird_Clips` and `Parsed_Not_Capuchinbird_Clips` are hardcoded in the script. You **must** use these exact folder names for your positive and negative samples, even if your target sound isn't a bird.
