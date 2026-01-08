# build_moves_pickle.py
"""
One time script to build a moves dictionary from PokeAPI, then save it as a local pickle.
# This is a data preparation utility: it creates a local dataset so training does not hit the API.

Run this ONCE (or rarely), not during training.
# Training should be offline and fast; the API is slow and rate limited.

Output formats:
- dex keyed: {1: [move_dicts], 2: [move_dicts], ...}
- name keyed: {"bulbasaur": [move_dicts], ...}
# Two possible dictionary keys: National Dex id or normalized Pokémon name.

Recommended: save name keyed, because your training code uses names.
# Your env and training sample matchups by Pokémon name, so name keyed saves you an extra mapping step.
"""

import argparse  # CLI arguments like --max_id and --out
import pickle    # Serialization format used to save the final move dictionary locally
import time      # Used to sleep between requests to be polite to the API
import requests  # HTTP client for calling PokeAPI endpoints


def extract_moves_full(pokemon_data, move_cache):
    """
    Convert raw PokeAPI pokemon JSON into a list of standardized move dicts.
    Keeps only physical/special moves (drops status moves).
    # This normalizes PokeAPI’s big nested structure into the small set of fields your simulator cares about.
    """
    out = []  # Output list of move dictionaries for this single Pokémon

    # PokeAPI provides a "moves" list with entries that contain move name and a URL to move details
    for entry in pokemon_data.get("moves", []):
        move_name = entry["move"]["name"]  # Raw move name string, e.g. "flamethrower"
        move_url = entry["move"]["url"]    # URL to fetch full move details (type, power, etc.)

        # Use a simple cache so if multiple Pokémon share the same move, we don’t refetch it every time
        if move_url in move_cache:
            move_data = move_cache[move_url]  # Reuse already downloaded move JSON
        else:
            move_data = requests.get(move_url, timeout=30).json()  # Fetch move details from PokeAPI
            move_cache[move_url] = move_data  # Store in cache for future Pokémon

        # Pull out only the fields your battle simulator uses
        move_type = (move_data.get("type") or {}).get("name")  # e.g. "fire"
        damage_class = (move_data.get("damage_class") or {}).get("name")  # "physical", "special", or "status"
        accuracy = move_data.get("accuracy")  # integer percent or None
        pp = move_data.get("pp")              # move PP (not currently used in your env, but saved)
        power = move_data.get("power")        # base power or None
        priority = move_data.get("priority")  # move priority (turn order modifier)

        # Keep only physical and special moves (status moves have no direct damage)
        # This matches your environment design where every move is assumed to be damage dealing.
        if damage_class not in ["physical", "special"]:
            continue  # Skip status moves like "growl" or "tail whip"

        # Store the move in a compact, consistent schema
        out.append(
            {
                "move": move_name,         # Move identifier
                "type": move_type,         # Elemental type used for effectiveness and STAB
                "damage_class": damage_class,  # Whether it uses Attack/Defense or Sp. Atk/Sp. Def
                "accuracy": accuracy,      # Used for hit/miss checks
                "pp": pp,                  # Saved for completeness / future mechanics
                "power": power,            # Used in the damage formula
                "priority": priority,      # Used in turn order resolution
            }
        )

    return out  # List of standardized move dicts for this Pokémon


def build_moves(max_id=493, sleep_s=0.05, keyed_by="name"):
    """
    Download pokemon 1..max_id from PokeAPI and build a dict of moves.
    keyed_by: 'name' or 'dex'
    # This loops through all Pokémon IDs in your chosen range and aggregates per Pokémon move lists.
    """
    move_cache = {}  # Global cache mapping move_url -> move_json to reduce duplicate move requests
    out = {}         # Final dictionary mapping either dex id or pokemon name -> list of moves

    # Iterate through the National Dex IDs you want to support
    for pid in range(1, int(max_id) + 1):
        url = f"https://pokeapi.co/api/v2/pokemon/{pid}"  # Endpoint for Pokémon base data
        data = requests.get(url, timeout=30).json()       # Fetch Pokémon JSON (includes move list)

        # Normalize the name so it matches your training code’s convention (lowercase, stripped)
        name = str(data.get("name", f"pokemon_{pid}")).strip().lower()

        # Convert PokeAPI move listing into your standardized move dicts using the shared cache
        moves = extract_moves_full(data, move_cache)

        # Store into output dictionary using the requested key style
        if keyed_by == "dex":
            out[int(pid)] = moves  # Key by integer dex id
        else:
            out[name] = moves      # Key by normalized name (recommended for your pipeline)

        # Periodic progress logging so you can see it’s working and how big the cache has grown
        if pid % 25 == 0:
            print(f"fetched {pid}/{max_id} | cache size: {len(move_cache)}")

        # Optional sleep to reduce rate limiting risk and be nice to the API
        if float(sleep_s) > 0.0:
            time.sleep(float(sleep_s))

    return out  # Full moves dictionary for all requested Pokémon


def main():
    # CLI wrapper: lets you configure the dataset build without editing code
    p = argparse.ArgumentParser()  # Create argument parser for command line usage
    p.add_argument("--max_id", type=int, default=493)  # Highest dex id to download (1..max_id)
    p.add_argument("--sleep_s", type=float, default=0.05)  # Delay between Pokémon requests
    p.add_argument("--keyed_by", choices=["name", "dex"], default="name")  # Output dict key scheme
    p.add_argument("--out", type=str, default="all_moves_by_name.pkl")  # Output pickle filepath
    args = p.parse_args()  # Parse CLI args into a simple namespace

    # Build the moves dictionary according to the chosen options
    moves = build_moves(max_id=args.max_id, sleep_s=args.sleep_s, keyed_by=args.keyed_by)

    # Write the result to disk as a pickle for fast loading during training
    with open(args.out, "wb") as f:
        pickle.dump(moves, f)

    # Print a final summary so you know what got saved
    print(f"saved pickle: {args.out} | entries: {len(moves)} | keyed_by: {args.keyed_by}")


if __name__ == "__main__":
    main()  # Standard Python entrypoint: only runs main() when executed as a script
