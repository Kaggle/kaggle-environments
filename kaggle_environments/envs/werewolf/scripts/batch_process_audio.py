import os
import sys
import glob
import argparse
import subprocess
import shutil
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Setup path for add_audio import
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
try:
    import add_audio
except ImportError:
    print("Could not import add_audio.py. Ensure it is in the same directory.")
    sys.exit(1)

def process_single_episode_direct(replay_file, bucket_base, config_path, tts_provider, prompt_path, cache_path, enable_llm, keep_temp=False, position=0):
    episode_id = os.path.splitext(os.path.basename(replay_file))[0]
    temp_out_dir = f"temp_audio_output/{episode_id}"
    
    # Ensure clean state
    if os.path.exists(temp_out_dir):
        shutil.rmtree(temp_out_dir)
    os.makedirs(temp_out_dir, exist_ok=True)
    
    success = False

    try:
        # 1. Generate Audio (Direct Python Call)
        # Pass TQDM kwargs for positioning
        tqdm_kwargs = {
            "position": position + 1, # Offset by 1 to leave room for main bar? Or main bar at 0?
             # Actually, if main bar is at 0, we use position N+1.
            "leave": False,
            "desc": f"Gen {episode_id}",
            "ncols": 80, # Limit width to avoid wrapping
            "mininterval": 0.5
        }
        
        # We need to capture stdout/stderr to separate logs from bars?
        # But add_audio prints to stdout/stderr.
        # Ideally we silence add_audio logging or redirect it.
        # But user wants bars. Bars print to stderr usually.
        
        add_audio.process_replay_file(
            input_path=replay_file,
            output_dir=temp_out_dir,
            config_path=config_path,
            tts_provider=tts_provider,
            prompt_path=prompt_path,
            cache_path=cache_path,
            disable_llm=not enable_llm,
            tqdm_kwargs=tqdm_kwargs
        )

        # Check if output directory has content
        # add_audio creates a subdirectory "audio" by default (standard.yaml)
        # We should check inside that validation.
        audio_subdir = os.path.join(temp_out_dir, "audio")
        
        if not os.path.isdir(audio_subdir):
             return False, f"No 'audio' subdirectory generated for {episode_id}"

        files = os.listdir(audio_subdir)
        if not files:
             return False, f"No audio files generated for {episode_id} (Audio dir empty)"
        
        wav_files = [f for f in files if f.endswith(".wav")]
        if len(wav_files) < 5:
             return False, f"Suspiciously low audio file count ({len(wav_files)}) for {episode_id}. Check logs."

        # 2. Upload to GCS
        target_path = f"{bucket_base}/{episode_id}"
        # Use gcloud storage rsync for robust directory synchronization
        upload_cmd = f"gcloud storage rsync '{temp_out_dir}' '{target_path}' --recursive"
        
        # We capture output to avoid spamming the console heavily, 
        # but maybe user wants to see upload progress too? 
        # For now, let's keep capture_output=True for upload to keep bars clean.
        upload_result = subprocess.run(
            upload_cmd, 
            shell=True, 
            capture_output=True, 
            text=True
        )
        
        if upload_result.returncode != 0:
             return False, f"Upload failed for {episode_id}: {upload_result.stderr}"

        success = True

    except Exception as e:
        return False, f"Exception in {episode_id}: {str(e)}"
    finally:
        # Cleanup ONLY if successful and not keeping temp
        if success and not keep_temp and os.path.exists(temp_out_dir):
            shutil.rmtree(temp_out_dir)
            
    return True, episode_id

def main():
    parser = argparse.ArgumentParser(description="Batch process audio generation and upload.")
    parser.add_argument("replay_dir", help="Directory containing .json replay files")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent workers (Limited for display)")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary audio files after upload")
    parser.add_argument("--log-file", type=str, default="batch_errors.log", help="Path to error log file")
    
    # Args expected by add_audio (defaults matching add_audio.py)
    parser.add_argument("-c", "--config_path", type=str,
                        default=os.path.join(script_dir, "configs/audio/standard.yaml"))
    parser.add_argument("--voice", type=str, default="gemini", choices=["chirp", "gemini"])
    parser.add_argument("--prompt_path", type=str,
                        default=os.path.join(script_dir, "configs/audio/theatrical_prompt.txt"))
    parser.add_argument("--cache_path", type=str, help="LLM cache file path.")
    parser.add_argument("--enable_llm_enhancement", action="store_true", help="Enable LLM enhancement (theatrical rewrites).")
    
    args = parser.parse_args()

    # Defaults
    if not args.cache_path:
        # We assume a shared cache for batch? Or per file?
        # Ideally shared cache if we want to save API calls across identical phrases (unlikely in different games).
        # Let's just use a default path in current dir.
        args.cache_path = "llm_cache.json"

    # Setup Logging
    logging.basicConfig(
        filename=args.log_file,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w' # Overwrite log file on new run
    )
    logger = logging.getLogger()

    all_files = glob.glob(os.path.join(args.replay_dir, "*.json"))
    # Filter for numeric IDs only (e.g. 74222013.json) to avoid processing summaries or other artifacts
    replay_files = [
        f for f in all_files 
        if os.path.basename(f).replace(".json", "").isdigit()
    ]
    if not replay_files:
        print(f"No json files found in {args.replay_dir}")
        return

    # Configuration
    bucket_base = "gs://kaggle-static/episode-assets/werewolf/episodes"
    
    print(f"Found {len(replay_files)} replays.")
    print(f"Processing with {args.workers} workers...")
    
    # IMPORTANT: 50 workers with progress bars might exceed terminal height.
    # We warn the user if workers > 20.
    if args.workers > 20:
        print("WARNING: High worker count might cause visual glitches with progress bars.")

    success_count = 0
    errors = []

    # ThreadPool Executor
    # We assign a static position index to each worker?
    # No, workers pick up tasks. We need to assign a slot (0..workers-1) to each running task.
    # We can use a Semaphore-guarded list of available slots.
    
    slot_lock = threading.Lock()
    # Initialize in reverse so pop() gives 0, 1, 2...
    available_slots = list(range(args.workers - 1, -1, -1))
    
    def get_slot():
        with slot_lock:
            return available_slots.pop()
            
    def release_slot(s):
        with slot_lock:
            available_slots.append(s)

    def worker_wrapper(file_path):
        slot = get_slot()
        try:
            return process_single_episode_direct(
                file_path, bucket_base, args.config_path, 
                args.voice, args.prompt_path, args.cache_path, 
                args.enable_llm_enhancement, args.keep_temp, 
                position=slot
            )
        finally:
            release_slot(slot)

    # Main Progress Bar (Overall)
    # position=args.workers to place it below all worker bars?
    # Or position=0 and shift workers down?
    # Let's put overall at position=0.
    # Workers at 1..N.
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {
            executor.submit(worker_wrapper, f): f 
            for f in replay_files
        }
        
        # position=0 is the main bar
        with tqdm(total=len(replay_files), desc="Total Progress", position=0, leave=True) as pbar:
            for future in as_completed(future_to_file):
                is_success, msg = future.result()
                if is_success:
                    success_count += 1
                else:
                    errors.append(msg)
                    logger.error(msg)
                    # We print errors to tqdm.write to avoid breaking layout, hopefully.
                    # tqdm.write(f"Error: {msg}") 
                    # If we use tqdm.write with many active bars, it might shift them. 
                    # Better to only log to file?
                    # User asked for error logs in file.
                pbar.update(1)

    print(f"\n\n\nCompleted. Success: {success_count}, Failed: {len(errors)}") # Newlines to clear bars
    
    if errors:
        print(f"\nErrors have been written to {args.log_file}")
        print("Check the log file for details.")

if __name__ == "__main__":
    main()
