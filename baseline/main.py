import audio_processing as ap
import os

if __name__ == "__main__":
    # Define folder with audio files and temp folder
    audio_folder = "audio_folder"
    temp_folder = "temp_chunks"
    
    # Ensure that the audio folder exists
    if not os.path.exists(audio_folder):
        print(f"Folder '{audio_folder}' does not exist.")
        exit(1)

    # Loop 1: Convert audio files to WAV format
    ap.ensure_wav_format(audio_folder)
    
    # Loop 2: Normalize volume
    #ap.normalize_audio_loudness(audio_folder)
    
    # Loop 3: Split audio into 1-2 min. chunks, isolate and transcribe voices
    ap.split_audio_into_chunks(audio_folder, temp_folder)

    corrected_folder = "corrected_transcriptions"

    # Loop 4: Apply spelling correction
    ap.correct_transcriptions(temp_folder, corrected_folder)
