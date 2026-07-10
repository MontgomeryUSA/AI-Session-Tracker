from hf_cache_setup import setup_hf_cache
from functions import * # import * means import everything
from openai import OpenAI
from datetime import datetime
import textwrap

#This accesses the local Ollama model the same way that we use Chat GPT
llm = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
#Huggin Face is a website that allows us to access AI models
#The first time we run the code we want to download and cache the model locally
#We need this token to tell HuggingFace that we are a member and have access to the 
#model that we want.
HF_TOKEN = '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'

#make sure whisper uses local models
#change offline to True once you get the models
setup_hf_cache("./models", offline=False)


#This file repesents one pass thru processing audio that we already have
audio_filename = './audio/conversation_10min.wav'

#Ensure audio is correct format
audio_filename = ensure_16k_mono(audio_filename)
#Transcribe and diarize
raw_path, proc_path = transcribe_whisperx(
    audio_file=audio_filename,
    hf_token=HF_TOKEN,
    use_gpu = True
)
#At this point we should have a _raw.json and a processed json with the transcription data
print("---Completed processing---")

'''
Now we have the data in a json file as nested objects.
WhisperX labels Speaker00 and Speaker01
It also can have consecutive objects for the same speaker like this:

{
    "speaker": "SPEAKER_01",
    "start": 9.665,
    "end": 14.472,
    "text": "Today we are going to be having a long English conversation because"
},
{
    "speaker": "SPEAKER_01",
    "start": 15.213,
    "end": 26.728,
    "text": "Over two years ago, back in February 2020, we made this conversation and millions of you loved this and said that it helped you to immerse yourself in English and have a fun time."
}

This should be merged into one entry
{
    "speaker": "SPEAKER_01",
    "start": 9.665,
    "end": 26.728,
    "text": "Today we are going to be having a long English conversation because Over two years ago, back in February 2020, we made this conversation and millions of you loved this and said that it helped you to immerse yourself in English and have a fun time."
}


'''

#remap speakers
speaker_map = {"SPEAKER_01": "Therapist", "SPEAKER_00": "Patient"}
#merge speakers with timestamps, need this for storing to SQL later
merged = merge_consecutive_speakers(proc_path, speaker_map)
#AI Summary does not need timestampes, strip them to save on tokens in the prompt
transcript_text = merged_to_prompt_text(merged)
print()
print(transcript_text)
print()

SYS = ("Draft a SOAP note ONLY from the transcript. Do not invent symptoms, "
      "diagnoses, or quotes. Mark anything inferred as [inferred]. Leave a "
      "field blank rather than guess.")

'''
#For now just respond with text but we may ask the model to give back a specific 
#json object so we can parse the data better. 

Return the following json object: 
{
 'subjective' : *fill in summary here*,
 'objective' : *fill in summary here*,
 'assessment' : *fill in summary here*,
 'plan' : *fill in summary here*,   
}
'''


print("Getting AI summary...")
stream = llm.chat.completions.create(
    model="gemma4:e2b",   # or deepseek-chat for prototyping
    #model="qwen3.5:4b", 
    messages=[
        {"role": "system", "content": SYS},
        {"role": "user", "content": transcript_text},
    ],
    stream = True
)

#---Stream the output to the terminal---
full_response = ""
#odd but you can go through chunks of the stream as they come back from the LLM
for chunk in stream:
    #this is based on the openai library and how it stores the chunk data
    data = chunk.choices[0].delta.content
    if data:
        print(data, end="", flush=True)
        full_response += data
print()

#--Save the summary to file--
# We might test this a bunch so put a timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
summary_filename = proc_path.stem.removesuffix("_proc") + f"_summary_{timestamp}.txt"
#store in the summary folder
summary_path = f"./summary/{summary_filename}"
#some of the lines were really long and hard to read, word wrap them
wrapped_response = wrap_text(full_response, width=80)

with open(summary_path, "w") as f:
    f.write(wrapped_response) 