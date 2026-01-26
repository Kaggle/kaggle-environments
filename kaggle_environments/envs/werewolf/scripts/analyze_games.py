import os
import sys
import json
import glob
import hashlib
import argparse
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Add current directory to path to import sibling scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import summarize_game
except ImportError:
    # Try creating absolute path import if running from root
    from kaggle_environments.envs.werewolf.scripts import summarize_game

def get_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def analyze_replays(replay_dir: str, cache_file: str, model_id: str, output_dir: Optional[str] = None) -> List[Dict]:
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            print(f"Loaded {len(cache)} cached analyses.")
        except Exception as e:
            print(f"Could not load cache: {e}. Starting fresh.")

    json_files = glob.glob(os.path.join(replay_dir, "*.json"))
    print(f"Found {len(json_files)} replay files in {replay_dir}.")

    results = []
    
    # Sort files to be deterministic
    json_files.sort()

    # Filter out the cache file itself if it's in the same dir and has .json extension
    json_files = [f for f in json_files if os.path.abspath(f) != os.path.abspath(cache_file)]

    for json_file in json_files:
        file_hash = get_file_hash(json_file)
        
        if file_hash in cache:
            # print(f"Skipping {os.path.basename(json_file)} (Cached)")
            cached_result = cache[file_hash]
            # Ensure filename is up to date in case it was renamed (optional, detailed)
            cached_result["_filename"] = os.path.basename(json_file)
            results.append(cached_result)

            if output_dir:
                base_name = os.path.splitext(os.path.basename(json_file))[0]
                summary_path = os.path.join(output_dir, f"{base_name}_summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(cached_result, f, indent=2)
                
                transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
                if not os.path.exists(transcript_path):
                     try:
                        t = summarize_game.extract_game_transcript(json_file)
                        with open(transcript_path, 'w') as f:
                            f.write(t)
                     except Exception as e:
                        print(f"Warning: Could not extract transcript for cached file {json_file}: {e}")

            continue
            
        print(f"Processing {os.path.basename(json_file)}...")
        transcript = summarize_game.extract_game_transcript(json_file)
        
        if "Error:" in transcript[:50] and len(transcript) < 200:
             print(f"Skipping {json_file}: {transcript}")
             continue

        if len(transcript) < 100:
            print(f"Skipping {json_file}: Transcript too short.")
            continue
            
        try:
            analysis = summarize_game.summarize_with_gemini(transcript, model_id=model_id)
            if analysis:
                # Convert pydantic to dict
                analysis_dict = analysis.model_dump()
                analysis_dict["_filename"] = os.path.basename(json_file)
                
                cache[file_hash] = analysis_dict
                results.append(analysis_dict)

                if output_dir:
                    base_name = os.path.splitext(os.path.basename(json_file))[0]
                    summary_path = os.path.join(output_dir, f"{base_name}_summary.json")
                    with open(summary_path, 'w') as f:
                        json.dump(analysis_dict, f, indent=2)
                    
                    transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
                    with open(transcript_path, 'w') as f:
                        f.write(transcript)
                
                # Save cache incrementally
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(cache, f, indent=2)
                except Exception as e:
                    print(f"Warning: Failed to write cache: {e}")
            else:
                 print(f"Failed to analyze {json_file} (LLM returned None)")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    return results

def generate_analysis_report(results: List[Dict], top_k: int = 5, report_file: str = "analysis_report.json"):
    if not results:
        print("No results to generate report from.")
        return

    # Filter out games without entertainment_metrics just in case
    valid_results = [r for r in results if r.get('entertainment_metrics')]
    
    # Sort games by excitement score
    sorted_games = sorted(valid_results, key=lambda x: x['entertainment_metrics']['excitement_score'], reverse=True)

    # 1. Top Games Overall
    top_games_overall = sorted_games[:top_k]

    # 2. Top Villager Wins
    villager_wins = [g for g in sorted_games if "villager" in g.get("winner_team", "").lower()]
    top_villager_wins = villager_wins[:top_k]

    # 3. Top Werewolf Wins
    werewolf_wins = [g for g in sorted_games if "werewolf" in g.get("winner_team", "").lower() or "werewolves" in g.get("winner_team", "").lower()]
    top_werewolf_wins = werewolf_wins[:top_k]

    # 4. Player Highlights (Stats & Best Games)
    player_data = defaultdict(lambda: {
        "games": 0,
        "mvp_count": 0,
        "best_game_overall": {"score": -1, "file": ""},
        "best_game_by_role": defaultdict(lambda: {"score": -1, "file": ""}),
        "stats": {"persuasion": [], "deception": [], "aggression": [], "analysis": []}
    })

    for game in valid_results:
        mvp = game.get('mvp_player', '')
        score = game['entertainment_metrics']['excitement_score']
        filename = game.get('_filename', 'Unknown')
        
        for p_stat in game.get('player_stats', []):
            name = p_stat['display_name']
            role = p_stat.get('role', 'Unknown')
            
            p_entry = player_data[name]
            p_entry["games"] += 1
            if name in mvp or mvp in name:
                p_entry["mvp_count"] += 1
            
            # Update Best Game Overall
            if score > p_entry["best_game_overall"]["score"]:
                p_entry["best_game_overall"] = {"score": score, "file": filename, "title": game.get('title', '')}
            
            # Update Best Game by Role
            if score > p_entry["best_game_by_role"][role]["score"]:
                p_entry["best_game_by_role"][role] = {"score": score, "file": filename, "title": game.get('title', '')}

            # Collect Stats
            p_entry["stats"]["persuasion"].append(p_stat.get('persuasion', 0))
            p_entry["stats"]["deception"].append(p_stat.get('deception', 0))
            p_entry["stats"]["aggression"].append(p_stat.get('aggression', 0))
            p_entry["stats"]["analysis"].append(p_stat.get('analysis', 0))

    # Construct JSON Report Structure
    report = {
        "config": {
            "top_k": top_k,
            "total_games_analyzed": len(valid_results)
        },
        "top_games_overall": [
            {
                "title": g.get('title'),
                "score": g['entertainment_metrics']['excitement_score'],
                "file": g.get('_filename'),
                "winner": g.get('winner_team'),
                "mvp": g.get('mvp_player')
            } for g in top_games_overall
        ],
        "top_villager_wins": [
            {
                "title": g.get('title'),
                "score": g['entertainment_metrics']['excitement_score'],
                "file": g.get('_filename'),
                "mvp": g.get('mvp_player')
            } for g in top_villager_wins
        ],
        "top_werewolf_wins": [
             {
                "title": g.get('title'),
                "score": g['entertainment_metrics']['excitement_score'],
                "file": g.get('_filename'),
                "mvp": g.get('mvp_player')
            } for g in top_werewolf_wins
        ],
        "player_highlights": {}
    }

    # Format Player Highlights
    for name, data in player_data.items():
        if data["games"] < 1: continue
        
        avg_stats = {
            k: sum(v)/len(v) if v else 0 for k, v in data["stats"].items()
        }
        
        report["player_highlights"][name] = {
            "games_played": data["games"],
            "mvp_count": data["mvp_count"],
            "average_stats": avg_stats,
            "best_game_overall": data["best_game_overall"],
            "best_game_by_role": dict(data["best_game_by_role"])
        }

    # Save JSON Report
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nAnalysis report saved to: {report_file}")
    except Exception as e:
        print(f"Error saving report: {e}")

    # Standard Output (Preserve existing terminal output style)
    print("\n" + "="*50)
    print(f"TOP {top_k} MOST ENTERTAINING GAMES")
    print("="*50)
    for i, g in enumerate(top_games_overall):
        metrics = g['entertainment_metrics']
        print(f"{i+1}. {g['title']} (Score: {metrics['excitement_score']}/10)")
        print(f"   Outcome: {metrics['outcome_type']}")
        print(f"   File: {g.get('_filename', 'Unknown')}")
        print(f"   MVP: {g.get('mvp_player', 'N/A')}")
        print("")

    print("\n" + "="*80)
    print("PLAYER AGGREGATE STATS (Min 1 games)")
    print("="*80)
    print(f"{'Player':<30} | {'Pers':<5} | {'Decp':<5} | {'Aggr':<5} | {'Anal':<5} | {'MVP':<3} | {'Games':<5}")
    print("-" * 80)
    
    sorted_players = sorted(player_data.items(), key=lambda x: x[1]['games'], reverse=True)
    for name, data in sorted_players:
         avg = {k: sum(v)/len(v) if v else 0 for k, v in data["stats"].items()}
         print(f"{name:<30} | {avg['persuasion']:5.1f} | {avg['deception']:5.1f} | {avg['aggression']:5.1f} | {avg['analysis']:5.1f} | {data['mvp_count']:<3} | {data['games']:<5}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_dir", help="Directory containing .json replays")
    parser.add_argument("--cache", default="analysis_cache.json", help="Path to cache file")
    parser.add_argument("--model", default="gemini-3-pro-preview", help="Gemini Model ID")
    parser.add_argument("--output-dir", help="Optional directory to save individual transcripts and summaries")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top games to list")
    parser.add_argument("--report-file", default="analysis_report.json", help="Path to save the JSON analysis report")
    
    args = parser.parse_args()
    
    results = analyze_replays(args.replay_dir, args.cache, args.model, args.output_dir)
    generate_analysis_report(results, args.top_k, args.report_file)
