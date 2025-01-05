import subprocess
import os
import torch
from pydub import AudioSegment
from huggingsound import SpeechRecognitionModel
from transformers import pipeline

def correct_transcriptions(temp_folder, corrected_folder):
    """
    Corrects the text in the transcription files and saves the results.
    """
    # Ensure that the corrected folder exists
    if not os.path.exists(corrected_folder):
        os.makedirs(corrected_folder)
        print(f"Corrected folder '{corrected_folder}' has been created.")
    
    # Download the spelling model
    corrector = pipeline("text2text-generation", model="oliverguhr/spelling-correction-german-base")

    # Edit all transcription files
    for file_name in os.listdir(temp_folder):
        if file_name.endswith("_transcription.txt"):
            input_file = os.path.join(temp_folder, file_name)
            output_file = os.path.join(corrected_folder, file_name.replace("_transcription", "_corrected"))

            print(f"Correct file: {file_name}...")
            with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
                for line in infile:
                    corrected_text = corrector(line.strip(), max_length=256)[0]["generated_text"]
                    outfile.write(corrected_text + "\n")
            print(f"Corrected file saved: {output_file}")

def ensure_wav_format(audio_folder):
    """
    Checks and converts audio files in the specified folder to WAV format.
    Files are replaced if they are not already in WAV format.
    """

    ignored_files = {".gitkeep"}  # Set of files to ignore

    for file_name in os.listdir(audio_folder):
        # Skip ignored files
        if file_name in ignored_files:
            print(f"Skipping ignored file: {file_name}")
            continue
        
        file_path = os.path.join(audio_folder, file_name)
        
        # Edit files only
        if os.path.isfile(file_path):
            # Check whether the file is already in WAV format
            if not file_name.lower().endswith(".wav"):
                print(f"Convert {file_name} to WAV...")
                
                # New file name with WAV extension
                new_file_path = os.path.splitext(file_path)[0] + ".wav"
                
                # FFmpeg command for converting to WAV format
                command = [
                    "ffmpeg", "-i", file_path,  # Input file
                    "-ar", "48000",             # Set sampling rate to 48 kHz
                    "-ac", "2",                 # Stereo channels
                    "-b:a", "192k",             # Bit rate for best quality
                    "-y",                       # Overwrite existing file
                    new_file_path
                ]
                
                # Execute FFmpeg
                print(f"Running command: {' '.join(command)}")
                subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                
                # Delete old file and rename new file
                os.remove(file_path)
                print(f"File {file_name} was successfully converted and replaced.")
            else:
                print(f"File {file_name} is already in WAV format. No conversion required.")

def normalize_audio_loudness(audio_folder):
    """
    Normalizes the volume of WAV files to -16 LUFS with FFmpeg.
    Files are replaced.
    """
    for file_name in os.listdir(audio_folder):
        file_path = os.path.join(audio_folder, file_name)
        
        # Edit WAV files only
        if file_name.lower().endswith(".wav") and os.path.isfile(file_path):
            print(f"Normalize volume of {file_name}...")
            
            # Temporary path for normalized file
            temp_file_path = os.path.splitext(file_path)[0] + "_normalized.wav"
            
            # FFmpeg command to normalize the volume
            command = [
                "ffmpeg", "-i", file_path,               # Input file
                "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",  # Normalization settings
                "-ar", "48000",                          # Set sampling rate to 48 kHz
                "-y",                                    # Overwrite existing file
                temp_file_path
            ]
            
            # Execute FFmpeg
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            
            # Delete old file and rename normalized file
            os.remove(file_path)
            os.rename(temp_file_path, file_path)
            print(f"Volume of {file_name} has been normalized and replaced.")

def split_audio_into_chunks(audio_folder, temp_folder, chunk_length_ms=120000):
    """
    Splits audio files from the audio folder into 1-2 min. chunks and saves them in the temp folder.
    The chunks are then processed with DeepFilterNet to isolate voices.
    The chunks are then transcribed with wav2vec and deleted.
    """
    # Ensure that the Temp folder exists
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
        print(f"Temp folder '{temp_folder}' was created.")
    
    # Load model and ensure GPU usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-german", device=device)

    for file_name in os.listdir(audio_folder):
        file_path = os.path.join(audio_folder, file_name)
        
        # Process only WAV files
        if file_name.lower().endswith(".wav") and os.path.isfile(file_path):
            print(f"Split file {file_name} into chunks...")
            
            # Load audio with pydub
            audio = AudioSegment.from_wav(file_path)
            
            # Split file into chunks
            for i, start_time in enumerate(range(0, len(audio), chunk_length_ms)):
                chunk = audio[start_time:start_time + chunk_length_ms]
                chunk_file_name = f"{os.path.splitext(file_name)[0]}_chunk_{i+1}.wav"
                chunk_file_path = os.path.join(temp_folder, chunk_file_name)
                
                # Store chunk
                chunk.export(chunk_file_path, format="wav")
                print(f"Chunk {i+1} saved: {chunk_file_path}")
                
                # Voice isolation with DeepFilterNet
                deepfilter_output = os.path.join(temp_folder, f"{os.path.splitext(chunk_file_name)[0]}_DeepFilterNet3.wav")
                # deepfilter generates names with additional attachment "_DeepFilterNet3"
                print(f"Isolate voices in {chunk_file_name} with DeepFilterNet...")
                
                command = [
                    "deepfilter", chunk_file_path,          # Input chunks
                    "-o", temp_folder                       # Output file
                ]
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    print(f"DeepFilterNet error: {result.stderr}")
                    continue
                
                # Replace the original chunk with the filtered chunk
                os.remove(chunk_file_path)
                os.rename(deepfilter_output, chunk_file_path)
                print(f"Voice isolation completed: {chunk_file_name}")
                
                # Transcription of the chunks with wav2vec
                print(f"Transcribe {chunk_file_name} with wav2vec...")
                print(f"Processing chunk on device: {next(model.model.parameters()).device}") #cuda:0 = GPU
                result = model.transcribe([chunk_file_path])
                transcription = result[0]['transcription']
                
                # Write transcription to a file
                transcription_file = os.path.join(temp_folder, f"{os.path.splitext(file_name)[0]}_transcription.txt")
                with open(transcription_file, "a", encoding="utf-8") as f:
                    f.write(f"Chunk {i+1}: {transcription}\n")
                print(f"Transcription saved: {transcription_file}")
                
                # Delete chunk
                os.remove(chunk_file_path)
                print(f"Temp file {chunk_file_name} deleted.")