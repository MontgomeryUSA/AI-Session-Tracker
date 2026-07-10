
from collections import Counter
import os
import subprocess
import soundfile as sf
from pathlib import Path
import json

#Target sampling rate and number of channels for WhisperX processing
TARGET_SR = 16000
TARGET_CH = 1

#Pass in an audio filename, read the audio file data and see 
#if it matches the format that we want. If it doesnt, convert the audio file
# with ffmpeg
def ensure_16k_mono(audio_filename, suffix="_16k"):
    info = sf.info(audio_filename)
    print(f"Input: {info.samplerate} Hz, {info.channels} ch")

    if info.samplerate == TARGET_SR and info.channels == TARGET_CH:
        print("Already 16 kHz mono, no conversion needed.")
        return audio_filename

    root, ext = os.path.splitext(audio_filename)
    out_path = f"{root}{suffix}{ext}"

    if os.path.exists(out_path):
        oi = sf.info(out_path)
        if oi.samplerate == TARGET_SR and oi.channels == TARGET_CH:
            print(f"Using existing converted file: {out_path}")
            return out_path

    print(f"Converting -> {out_path} ...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_filename,
         "-ac", str(TARGET_CH), "-ar", str(TARGET_SR),
         "-sample_fmt", "s16", out_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
        
    print("Conversion done.")
    return out_path

#Whisper creates a json file that has a nested object for each individual word
# we dont need that right now so process and remove
def process_whisper_output(json_path : Path):
    #json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_segments = []

    for seg in data["segments"]:
        words = seg.get("words", [])

        speakers = [
            w["speaker"]
            for w in words
            if "speaker" in w and w["speaker"] is not None
        ]

        if speakers:
            speaker = Counter(speakers).most_common(1)[0][0]
        else:
            speaker = "UNKNOWN"

        processed_segments.append({
            "speaker": speaker,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })

    output_data = {
        "segments": processed_segments
    }

    output_path = json_path.with_name(
        json_path.stem.removesuffix("_raw") + "_proc.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    return str(output_path)

#This runs the WhisperX library on audio to get transcription and diarization data
def transcribe_whisperx(
    audio_file: str,
    hf_token: str,
    model: str = "small",
    min_speakers: int = 2,
    max_speakers: int = 2,
    use_gpu: bool = False
):
    import whisperx  # Import here to ensure env vars are set
    audio_path = Path(audio_file).resolve()
    
    #check if the files exist so we dont keep reconverting on all of our tests
    raw_path = audio_path.with_name(f"{audio_path.stem}_raw.json")
    proc_path = audio_path.with_name(f"{audio_path.stem}_proc.json")
    json_path = audio_path.with_name(f"{audio_path.stem}.json")
    #direct all of them to the transcript folder
    raw_path = audio_path.parent.parent / "transcript" / f"{audio_path.stem}_raw.json"
    proc_path = audio_path.parent.parent / "transcript" / f"{audio_path.stem}_proc.json"
    json_path = audio_path.parent.parent / "transcript" / f"{audio_path.stem}.json"

    
    print(f"{proc_path}")
    print(f"{raw_path}")
    print(f"{json_path}")
    #return ""

    if raw_path.exists() and proc_path.exists():
        print(f"Found both files, whisperx already processed { audio_path}")
        return raw_path, proc_path
    
    if not raw_path.exists():
        cmd = [
            "whisperx",
            str(audio_path),
            #"--model", model,   #download and try the small model
            
            #look at --model_cache_only       
            "--diarize",
            "--hf_token", hf_token,
            "--min_speakers", str(min_speakers),
            "--max_speakers", str(max_speakers),
            "--output_format", "json",
            "--output_dir", str(json_path.parent),
        ]
        
        if use_gpu:
            cmd.extend(["--compute_type", "float32"])
        else:
            cmd.extend([
                "--device", "cpu",
                "--compute_type", "int8"
            ])
            
        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="")

        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                f"WhisperX failed with code {return_code}"
            )

        #Rename the file because we will be processing it        
        transcript_dir = json_path.parent.parent / "transcript"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        raw_path = transcript_dir / f"{json_path.stem}_raw{json_path.suffix}"
        json_path.rename(raw_path)
    else:
        print(f"Skipping whisperx processing...found {json_path}")
    
    if not proc_path.exists():
        print(f"Processing the whisperx raw json file: {raw_path}")
        #process the file to remove nested objects
        json_path = process_whisper_output(raw_path)
    

    return raw_path, proc_path

## WhisperX had consecutive speaker objects, we can combine
def merge_consecutive_speakers(proc_path : Path, speaker_map=None):
    """Merge consecutive same-speaker segments into blocks.
    Returns list of {speaker, text, start, end} — keep this for storage,
    strip timestamps only when building the LLM prompt."""
    
    data = None
    
    with open(proc_path) as f:
        data = json.load(f)
        
    if data == None:
        print(f"Error merging from path: {proc_path}")
        return
    
    segments = data["segments"]
    speaker_map = speaker_map or {}
    merged = []
    for seg in segments:
        speaker = speaker_map.get(seg["speaker"], seg["speaker"])
        text = seg["text"].strip()
        if merged and merged[-1]["speaker"] == speaker:
            merged[-1]["text"] += " " + text
            merged[-1]["end"] = seg["end"]          # extend to this segment's end
        else:
            merged.append({
                "speaker": speaker,
                "text": text,
                "start": seg["start"],               # first segment's start
                "end": seg["end"],
            })
    return merged

# AI prompt doesnt need timestamps, remove
def merged_to_prompt_text(merged):
    """LLM input: no timestamps, just speaker-labeled lines."""
    return "\n".join(f"{m['speaker']}: {m['text']}" for m in merged)


## -- Wrapping Functions ##
## The AI was returning incredible long lines, hard to read, reformat##
def wrap_line(line, width=80):
    wrapped_lines = []

    while len(line) > width:
        # Look for the last space before width
        split_at = line.rfind(" ", 0, width + 1)

        # If no space found, hard split at width
        if split_at == -1:
            split_at = width

        wrapped_lines.append(line[:split_at].rstrip())
        line = line[split_at:].lstrip()

    wrapped_lines.append(line)
    return wrapped_lines


def wrap_text(text, width=80):
    output_lines = []

    for line in text.splitlines():
        if len(line) > width:
            output_lines.extend(wrap_line(line, width))
        else:
            output_lines.append(line)

    return "\n".join(output_lines)

#Not used??
def transcript_to_prompt_text(json_path):
    with open(json_path) as f:
        data = json.load(f)
    lines = []
    for seg in data["segments"]:
        speaker = "Therapist" if seg["speaker"] == "SPEAKER_01" else "Patient"  # map once you know which is which
        lines.append(f"{speaker}: {seg['text'].strip()}")
    return "\n".join(lines)