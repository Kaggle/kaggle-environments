import os
import sys
import glob
import argparse
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_single_episode(replay_file, bucket_base, script_path):
    episode_id = os.path.splitext(os.path.basename(replay_file))[0]
    temp_out_dir = f"temp_audio_output/{episode_id}"
    
    # Ensure clean state
    if os.path.exists(temp_out_dir):
        shutil.rmtree(temp_out_dir)
    os.makedirs(temp_out_dir, exist_ok=True)
    
    try:
        # 1. Generate Audio
        # We disable LLM enhancement for speed if desired, matching the original script
        cmd = [
            sys.executable, script_path,
            "-i", replay_file,
            "-o", temp_out_dir,
            "--disable_llm_enhancement",
            "--quiet" # Adding quiet flag if supported or just relying on capture_output
        ]
        
        # We capture output to avoid spamming the console
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return False, f"Audio check failed for {episode_id}: {result.stderr}"

        # 2. Upload to GCS
        target_path = f"{bucket_base}/{episode_id}/"
        upload_cmd = [
            "gsutil", "-m", "cp", "-r", 
            f"{temp_out_dir}/*", 
            target_path
        ]
        # Shell=True might be needed for wildcard expansion in arguments if passed as string, 
        # but with list args, wildcard * inside string doesn't expand. 
        # Better to iterate files or use shell=True carefully.
        # Actually simplest is: gsutil -m cp -r temp_dir/* gs://...
        # But subprocess w/ list doesn't expand *. 
        # We'll just upload the directory content by uploading the directory itself?
        # gsutil cp -r dir gs://bucket/path/ -> gs://bucket/path/dir/... which might be wrong nesting.
        # The original script did: gsutil -m cp -r "$TEMP_OUT_DIR"/* "$TARGET_PATH"
        
        # Let's use shell=True for the upload command to support wildcard
        upload_result = subprocess.run(
            f"gsutil -m cp -r '{temp_out_dir}'/* '{target_path}'", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        
        if upload_result.returncode != 0:
             return False, f"Upload failed for {episode_id}: {upload_result.stderr}"

    except Exception as e:
        return False, str(e)
    finally:
        # Cleanup
        if os.path.exists(temp_out_dir):
            shutil.rmtree(temp_out_dir)
            
    return True, episode_id

def main():
    parser = argparse.ArgumentParser(description="Batch process audio generation and upload.")
    parser.add_argument("replay_dir", help="Directory containing .json replay files")
    parser.add_argument("--workers", type=int, default=20, help="Number of concurrent workers")
    args = parser.parse_args()

    replay_files = glob.glob(os.path.join(args.replay_dir, "*.json"))
    if not replay_files:
        print(f"No json files found in {args.replay_dir}")
        return

    # Configuration
    bucket_base = "gs://kaggle-static/episode-visualizers/werewolf/default/audio"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming add_audio.py is in the same directory as this script
    add_audio_script = os.path.join(script_dir, "add_audio.py")
    
    print(f"Found {len(replay_files)} replays.")
    print(f"Processing with {args.workers} workers...")
    
    success_count = 0
    errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {
            executor.submit(process_single_episode, f, bucket_base, add_audio_script): f 
            for f in replay_files
        }
        
        with tqdm(total=len(replay_files), desc="Overall Progress") as pbar:
            for future in as_completed(future_to_file):
                is_success, msg = future.result()
                if is_success:
                    success_count += 1
                else:
                    errors.append(msg)
                pbar.update(1)

    print(f"\nCompleted. Success: {success_count}, Failed: {len(errors)}")
    if errors:
        print("\nErrors:")
        for e in errors[:20]: # Show first 20 errors
            print(e)
        if len(errors) > 20:
            print(f"... and {len(errors)-20} more.")

if __name__ == "__main__":
    main()
