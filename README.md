# Natural Language Processing (NLP) - Hugging Face Wac2Vec German - transcription pipeline
[![License MIT](https://img.shields.io/github/license/l4nz8/q_play)](https://opensource.org/licenses/MIT)
[![Linting](https://github.com/l4nz8/q_play/actions/workflows/main.yml/badge.svg)](https://github.com/l4nz8/q_play/actions/workflows/main.yml)

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

## Introduction 🚀

**nlp-ger_hf** is a complete audio transcription pipeline for German language processing. The workflow:
1. Converts audio files into high-quality **WAV format**.
2. Normalizes audio loudness to **-16 LUFS**.
3. Splits audio into **1-2 minute chunks**.
4. Applies **DeepFilterNet** to isolate voices.
5. Transcribes chunks using **wav2vec2**.
6. Corrects transcriptions with a **spelling correction model** for clarity.

---

## Table of Contents 📖
1. [Introduction](#introduction-)
2. [Features](#features)
3. [Requirements](#requirements-)
4. [Download and Install](#download-and-install)
5. [Usage](#usage)
6. [Dependencies](#dependencies)
7. [License](#license)
8. [Contributions](#contributions)

---

## Features
- High-quality audio conversion using **FFmpeg**.
- Voice isolation via **DeepFilterNet**.
- Accurate German transcription using **wav2vec**.
- Automatic spelling and grammar correction with HuggingFace models.

## Requirements 💻

### Tools
- [FFmpeg](https://ffmpeg.org/download.html) – Required for audio conversion.
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) – For voice isolation.
#### Recommended:
- [CUDA](https://developer.nvidia.com/cuda-toolkit) - To be able to load the models onto the `GPU` and process the audio files faster.

Note: `CUDA` must be installed manually for your individual `GPU`.


## Download and Install
🐍 Python 3.9 is recommended. Other versions may work but have not been tested.

1. **Clone the Repository:**
```bash
git clone https://github.com/l4nz8/nlp-ger_hf.git
```
2. **Install Python Dependencies:**
```bash
cd nlp-ger_hf
pip3 install -r requirements.txt
```
3. **Verify FFmpeg Installation:**
```bash
ffmpeg -version
```
4. **Set Up DeepFilterNet:** Follow the installation instructions [here](https://github.com/Rikorose/DeepFilterNet).

## Usage
1. **Input Folder:** Place your audio files (e.g., `.mp3`, `.wav`) into the `audio_files` folder.

2. **Run the Workflow:**

```bash
python baseline/main.py
```
3. **Output:**
- Transcriptions: Stored in temp_chunks/ as text files.
- Corrected Transcriptions: Saved in corrected_transcriptions/.
### Workflow Overview
<table>
  <thead>
    <tr>
      <th>Step</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>Convert audio files to WAV format</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Normalize audio loudness to -16 LUFS</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Split audio into 1-2 minute chunks</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Isolate voices using DeepFilterNet</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Transcribe audio using wav2vec2</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Correct transcriptions using HuggingFace</td>
    </tr>
  </tbody>
</table>

### Folder Structure
```bash
nlp-ger_hf/
│
├── audio_files/                # Input audio files
├── temp_chunks/                # Temporary processed chunks
├── corrected_transcriptions/   # Corrected transcription output
├── baseline/
│   └── main.py                 # Main script
└── requirements.txt            # Python dependencies
```

### Notes:
1. **Directory Comments**: Comments clarify the purpose of each folder and file.
2. **`baseline/`**: Indicates the directory containing your `main.py` script.
3. **Copy-Paste Ready**: You can copy this into your Markdown file, and it will render correctly in VS Code, GitHub, or any Markdown viewer. 

If you need further tweaks, let me know! 🚀

### Example Workflow
**Input:**
- `speech.mp3` (placed in `audio_files`)

**Output:**
- **Transcribed chunks:**
    - speech_chunk_1_transcription.txt
    - speech_chunk_2_transcription.txt
- **Corrected file:**
    - `speech_corrected.txt` (saved in `corrected_transcriptions/`)

## Dependencies
- **wav2vec2:** `jonatasgrosman/wav2vec2-large-xlsr-53-german` (for transcription).
- **Spelling Correction:** `oliverguhr/spelling-correction-german-base`.

## License
This project is licensed under the [MIT License](https://opensource.org/license/MIT).

## Contributions
Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit a pull request.