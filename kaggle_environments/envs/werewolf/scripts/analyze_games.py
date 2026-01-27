import os
import sys
import json
import glob
import hashlib
import argparse
import csv
from typing import Dict, List, Any, Optional
from collections import defaultdict
from tqdm import tqdm
from pydantic import BaseModel, Field

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

class GameReport(BaseModel):
    title: str
    score: float
    file: str
    mvp: str
    mvp_role: str
    mvp_reasoning: str
    winner: str
    total_turns: int
    rubric: Dict[str, Any]




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

def analyze_single_game(json_file: str, cache: Dict, model_id: str, output_dir: Optional[str], max_retries: int = 10) -> Optional[Dict]:
    """
    Analyzes a single game replay.
    Returns the analysis dict (either from cache or fresh) or None if failed.
    """
    try:
        file_hash = get_file_hash(json_file)
        
        # Check in-memory cache first (though typical usage expects cache populated initially)
        if file_hash in cache:
             # print(f"Skipping {os.path.basename(json_file)} (Cached)")
             cached_result = cache[file_hash]
             cached_result["_filename"] = os.path.basename(json_file)
             return cached_result, file_hash, False
        
        # Check if summary json exists in output_dir
        if output_dir:
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            summary_path = os.path.join(output_dir, f"{base_name}_summary.json")
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r') as f:
                        existing_analysis = json.load(f)
                    # We assume if the summary exists, it's valid. 
                    # We might want to backfill _filename if missing
                    existing_analysis["_filename"] = os.path.basename(json_file)
                    # Update cache with this existing result so next run is faster
                    # Note: We won't have the compute hash if we just load json, 
                    # but we computed file_hash above.
                    return existing_analysis, file_hash, True 
                except Exception as e:
                    print(f"Warning: Failed to load existing summary {summary_path}: {e}")

        # If not in cache and not on disk, proceed with analysis

        print(f"Processing {os.path.basename(json_file)}...")
        transcript, turn_count = summarize_game.extract_game_transcript(json_file)

        if "Error:" in transcript[:50] and len(transcript) < 200:
                print(f"Skipping {json_file}: {transcript}")
                return None, file_hash, False

        if len(transcript) < 100:
            print(f"Skipping {json_file}: Transcript too short.")
            return None, file_hash, False
            
        analysis = summarize_game.summarize_with_gemini(transcript, model_id=model_id, max_retries=max_retries)
        if analysis:
            analysis.total_turns = turn_count
            analysis_dict = analysis.model_dump()
            analysis_dict["_filename"] = os.path.basename(json_file)
            
            if output_dir:
                base_name = os.path.splitext(os.path.basename(json_file))[0]
                summary_path = os.path.join(output_dir, f"{base_name}_summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(analysis_dict, f, indent=2)
                
                transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
                with open(transcript_path, 'w') as f:
                    f.write(transcript)
            
            return analysis_dict, file_hash, True
        else:
                print(f"Failed to analyze {json_file} (LLM returned None)")
                return None, file_hash, False

    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return None, None, False

def analyze_replays(replay_dir: str, cache_file: str, model_id: str, output_dir: Optional[str] = None, max_workers: int = 20, max_retries: int = 10) -> List[Dict]:
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
    
    # Sort files to be deterministic
    json_files.sort()

    # Filter out the cache file itself
    json_files = [f for f in json_files if os.path.abspath(f) != os.path.abspath(cache_file)]
    
    results = []
    new_entries_count = 0
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    print(f"Starting analysis with {max_workers} workers...")
    
    
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        # Submit all tasks
        future_to_file = {
            executor.submit(analyze_single_game, f, cache, model_id, output_dir, max_retries): f 
            for f in json_files
        }
        
        with tqdm(total=len(json_files), desc="Analyzing Games") as pbar:
            for future in as_completed(future_to_file):
                json_file = future_to_file[future]
                try:
                    result, file_hash, is_new = future.result()
                    if result:
                        results.append(result)
                        if is_new and file_hash:
                            cache[file_hash] = result
                            new_entries_count += 1
                except Exception as e:
                    print(f"Exception analyzing {json_file}: {e}")
                finally:
                    pbar.update(1)
                
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user. Shutting down workers...")
        executor.shutdown(wait=False, cancel_futures=True)
        # Still try to save whatever we have cached so far
        if new_entries_count > 0:
             print(f"Saving {new_entries_count} new entries before exit...")
             try:
                with open(cache_file, 'w') as f:
                    json.dump(cache, f, indent=2)
             except: pass
        sys.stdout.flush()
        # Force exit to kill non-daemon threads from file executor
        os._exit(1)
    finally:
        # multiple calls to shutdown are safe
        executor.shutdown(wait=False, cancel_futures=True)

    # Save updated cache at the end if we added anything
    if new_entries_count > 0:
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f"Updated cache with {new_entries_count} new entries.")
        except Exception as e:
            print(f"Warning: Failed to write cache: {e}")
    else:
        print("No new analyses to cache.")

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
        "stats": {"persuasion": [], "deception": [], "aggression": [], "analysis": []},
        "rubric_sums": defaultdict(list),
        "all_games": []
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
            
            # Collect Game Rubrics for aggregation
            rubric = game['entertainment_metrics'].get('rubric', {})
            for r_key, r_val in rubric.items():
                p_entry["rubric_sums"][r_key].append(r_val)
                
            # Store game info for top-k lists
            game_mvp_role = "Unknown"
            if mvp != 'N/A' and mvp:
                for p in game.get('player_stats', []):
                    if p.get('display_name') == mvp:
                        game_mvp_role = p.get('role', 'Unknown')
                        break
            
            game_info = {
                "score": score,
                "file": filename,
                "title": game.get('title', ''),
                "rubric": rubric,
                "winner": game.get('winner_team', 'Unknown'),
                "mvp": mvp,
                "mvp_role": game_mvp_role,
                "mvp_reasoning": game.get('mvp_reasoning', ''),
                "total_turns": game.get('total_turns', 0),
                "role": role # identifying role for this player's perspective (not in schema but needed for filtering)
            }
            p_entry["all_games"].append(game_info)

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

    # Helper to get mvp role
    def get_mvp_role(game: Dict) -> str:
        mvp = game.get('mvp_player', 'N/A')
        if mvp == 'N/A': return "N/A"
        for p in game.get('player_stats', []):
            if p.get('display_name') == mvp:
                return p.get('role', 'Unknown')
        return "Unknown"

    def to_game_report(g: Dict, from_processed: bool = False) -> Dict:
        if from_processed:
            # 'g' is game_info from player_data
            return GameReport(
                title=g['title'],
                score=g['score'],
                file=g['file'],
                mvp=g['mvp'],
                mvp_role=g['mvp_role'],
                mvp_reasoning=g['mvp_reasoning'],
                winner=g['winner'],
                total_turns=g['total_turns'],
                rubric=g['rubric']
            ).model_dump()
        else:
            # 'g' is raw game dict
            return GameReport(
                title=g.get('title', ''),
                score=g['entertainment_metrics']['excitement_score'],
                file=g.get('_filename', ''),
                mvp=g.get('mvp_player', 'N/A'),
                mvp_role=get_mvp_role(g),
                mvp_reasoning=g.get('mvp_reasoning', ''),
                winner=g.get('winner_team', 'Unknown'),
                total_turns=g.get('total_turns', 0),
                rubric=g['entertainment_metrics'].get('rubric', {})
            ).model_dump()

    # Construct JSON Report Structure
    report = {
        "config": {
            "top_k": top_k,
            "total_games_analyzed": len(valid_results)
        },
        "top_games_overall": [to_game_report(g, False) for g in top_games_overall],
        "top_villager_wins": [to_game_report(g, False) for g in top_villager_wins],
        "top_werewolf_wins": [to_game_report(g, False) for g in top_werewolf_wins],
        "player_highlights": {}
    }

    # Format Player Highlights
    for name, data in player_data.items():
        if data["games"] < 1: continue
        
        avg_stats = {
            k: sum(v)/len(v) if v else 0 for k, v in data["stats"].items()
        }
        
        avg_rubrics = {
            k: sum(v)/len(v) if v else 0 for k, v in data["rubric_sums"].items()
        }
        
        # Helper to check if player won and was MVP
        def is_mvp_win(name, g):
            # Check 1: Is Player MVP?
            if g['mvp'] != name:
                 return False

            # Check 2: Did they win?
            winner = g['winner'].lower()
            role_lower = g['role'].lower()
            won = False
            
            if "werewolf" in role_lower:
                if "werewolf" in winner or "werewolves" in winner:
                    won = True
            else:
                # Villager, Seer, Doctor
                if "villager" in winner:
                    won = True
            
            return won

        # Filter all games for this player to only include MVP wins
        mvp_wins = [g for g in data["all_games"] if is_mvp_win(name, g)]
        
        # Sort by score
        mvp_wins_sorted = sorted(mvp_wins, key=lambda x: x['score'], reverse=True)
        
        # Process Top K games (Overall)
        top_games_overall_list = mvp_wins_sorted[:top_k]
        
        # Process Top K by Role
        games_by_role = defaultdict(list)
        for g in mvp_wins_sorted:
            games_by_role[g['role']].append(g)
            
        top_games_by_role_list = {
            role: games[:top_k] for role, games in games_by_role.items()
        }
        
        # Update Best Game Overall (derived from filtered list)
        best_game_overall = mvp_wins_sorted[0] if mvp_wins_sorted else None
        
        # Update Best Game by Role (derived from filtered list)
        best_game_by_role = {}
        for role, games in games_by_role.items():
            if games:
                best_game_by_role[role] = games[0]

        report["player_highlights"][name] = {
            "games_played": data["games"],
            "mvp_count": data["mvp_count"],
            "average_stats": avg_stats,
            "average_rubrics": avg_rubrics,
            "top_games_overall": [to_game_report(g, True) for g in top_games_overall_list],
            "top_games_by_role": {r: [to_game_report(g, True) for g in games] for r, games in top_games_by_role_list.items()},
            "best_game_overall": to_game_report(best_game_overall, True) if best_game_overall else None, 
            "best_game_by_role": {r: to_game_report(g, True) for r, g in best_game_by_role.items()}
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
        mvp_name = g.get('mvp_player', 'N/A')
        mvp_role = "Unknown"
        if mvp_name != 'N/A':
            for p in g.get('player_stats', []):
                if p.get('display_name') == mvp_name:
                    mvp_role = p.get('role', 'Unknown')
                    break

        print(f"{i+1}. {g['title']} (Score: {metrics['excitement_score']:.1f}/10)")
        print(f"   Winner: {g.get('winner_team', 'Unknown')}")
        print(f"   Outcome: {metrics['outcome_type']}")
        
        # Display Rubric if available (backward compatibility check)
        if 'rubric' in metrics:
            r = metrics['rubric']
            print(f"   Rubric: Strat:{r.get('strategic_depth')} | Unpred:{r.get('unpredictability')} | Nar:{r.get('narrative_quality')} | Skill:{r.get('player_competence')} | Pace:{r.get('pacing')} | Humor:{r.get('humor')} | Subj:{r.get('subjective_impression')} | Syn:{r.get('synergy', '-')}")
        
        print(f"   File: {g.get('_filename', 'Unknown')}")
        print(f"   MVP: {mvp_name} ({mvp_role})")
        print("")

    print("\n" + "="*80)
    print("PLAYER AGGREGATE STATS (Min 1 games)")
    print("="*80)
    print(f"{'Player':<30} | {'Pers':<5} | {'Decp':<5} | {'Aggr':<5} | {'Anal':<5} | {'MVP':<3} | {'Games':<5}")
    print("-" * 80)
    
    sorted_players = sorted(player_data.items(), key=lambda x: x[1]['games'], reverse=True)
    # Filter out players with fewer than 10 games to remove noise/default IDs from failed runs
    visible_players = [p for p in sorted_players if p[1]['games'] >= 10]

    for name, data in visible_players:
         avg = {k: sum(v)/len(v) if v else 0 for k, v in data["stats"].items()}
         print(f"{name:<30} | {avg['persuasion']:5.1f} | {avg['deception']:5.1f} | {avg['aggression']:5.1f} | {avg['analysis']:5.1f} | {data['mvp_count']:<3} | {data['games']:<5}")

def save_csv_report(results: List[Dict], csv_file: str):
    """Saves a CSV report with rubric scores, MVP, and key metrics."""
    if not results:
        print("No results to save to CSV.")
        return

    fieldnames = [
        "File", "Title", "Score", "Winner", "Outcome", "Turns",
        "Strategic Depth", "Unpredictability", "Narrative Quality", 
        "Humor", "Player Competence", "Pacing", "Subjective Impression", "Synergy",
        "MVP", "MVP Role", "MVP Reasoning"
    ]

    try:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for res in results:
                metrics = res.get('entertainment_metrics', {})
                rubric = metrics.get('rubric', {})
                
                # Find MVP Role
                mvp_name = res.get('mvp_player', 'N/A')
                mvp_role = "Unknown"
                if mvp_name != 'N/A':
                    for p in res.get('player_stats', []):
                        if p.get('display_name') == mvp_name:
                            mvp_role = p.get('role', 'Unknown')
                            break

                row = {
                    "File": res.get("_filename", "Unknown"),
                    "Title": res.get("title", ""),
                    "Score": metrics.get("excitement_score", 0),
                    "Winner": res.get("winner_team", "Unknown"),
                    "Outcome": metrics.get("outcome_type", ""),
                    "Turns": res.get("total_turns", 0),
                    "Strategic Depth": rubric.get("strategic_depth", 0),
                    "Unpredictability": rubric.get("unpredictability", 0),
                    "Narrative Quality": rubric.get("narrative_quality", 0),
                    "Humor": rubric.get("humor", 0),
                    "Player Competence": rubric.get("player_competence", 0),
                    "Pacing": rubric.get("pacing", 0),
                    "Subjective Impression": rubric.get("subjective_impression", 0),
                    "Synergy": rubric.get("synergy", 0),
                    "MVP": mvp_name,
                    "MVP Role": mvp_role,
                    "MVP Reasoning": res.get("mvp_reasoning", "")
                }
                writer.writerow(row)
        print(f"CSV report saved to: {csv_file}")
    except Exception as e:
        print(f"Error saving CSV report: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_dir", help="Directory containing .json replays")
    parser.add_argument("--cache", default="analysis_cache.json", help="Path to cache file")
    parser.add_argument("--model", default="gemini-3-pro-preview", help="Gemini Model ID")
    parser.add_argument("--output-dir", help="Optional directory to save individual transcripts and summaries")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top games to list")
    parser.add_argument("--report-file", default="analysis_report.json", help="Path to save the JSON analysis report")
    parser.add_argument("--max-workers", type=int, default=50, help="Max parallel workers for analysis")
    parser.add_argument("--max-retries", type=int, default=10, help="Max retries for API quota errors")
    parser.add_argument("--csv-file", help="Path to save the CSV analysis report")

    args = parser.parse_args()

    results = analyze_replays(args.replay_dir, args.cache, args.model, args.output_dir, args.max_workers, args.max_retries)
    generate_analysis_report(results, args.top_k, args.report_file)
    
    if args.csv_file:
        save_csv_report(results, args.csv_file)
