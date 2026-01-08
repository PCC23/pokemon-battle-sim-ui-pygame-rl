# data.py
"""
Loads and serves all local datasets needed by the environment + training.

This module is intentionally "boring":
- NO web requests
- Just reads CSV + pickle once, exposes helper getters/functions
"""

import pickle
import pandas as pd

# Module level globals other modules import/use
df_poke = None            # pandas DataFrame of base stats + types
type_chart_df = None      # pandas DataFrame of type multipliers
type_chart = None         # type_chart_df indexed for fast lookup
all_moves_by_name = None  # dict: pokemon_name_lower -> list[move_dict]


def load_data(
    pokemon_csv_path="pokemon_kanto_johto_sinnoh.csv",
    type_chart_csv_path="type_chart.csv",
    moves_pickle_path="all_moves_by_pokemon.pkl",
):
    """
    Load all datasets ONCE and store in globals.

    Expected files:
    - pokemon_csv_path: must contain at least columns like:
      Name, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, Type1, Type2 (Type2 can be empty)
    - type_chart_csv_path: must contain a column 'Attacking' and columns for defending types
    - moves_pickle_path: can be either:
        A) {dex_int: [move_dict, ...]}  (older format)
        B) {name_str: [move_dict, ...]} (preferred)
      This function will normalize to all_moves_by_name[name_lower] = moves list.
    """
    global df_poke, type_chart_df, type_chart, all_moves_by_name

    # 1) Pokémon base stats + types
    df_poke = pd.read_csv(pokemon_csv_path).reset_index(drop=True)
    df_poke["id"] = df_poke.index + 1

    # 2) Type chart
    type_chart_df = pd.read_csv(type_chart_csv_path)
    if "Attacking" not in type_chart_df.columns:
        raise ValueError("type_chart.csv must contain an 'Attacking' column.")

    # Make fast lookup table: row = attacking type, col = defending type
    type_chart = type_chart_df.set_index("Attacking")

    # 3) Moves dictionary (pickle)
    with open(moves_pickle_path, "rb") as f:
        moves_obj = pickle.load(f)

    # Normalize moves pickle to name -> moves
    all_moves_by_name = _normalize_moves_dict(moves_obj)

    # Fail fast sanity checks
    if df_poke is None or len(df_poke) == 0:
        raise ValueError("df_poke is empty. Check pokemon_csv_path.")
    if type_chart_df is None or len(type_chart_df) == 0:
        raise ValueError("type_chart_df is empty. Check type_chart_csv_path.")
    if type_chart is None or len(type_chart.index) == 0:
        raise ValueError("type_chart is empty after indexing. Check type_chart_csv_path.")
    if all_moves_by_name is None or len(all_moves_by_name) == 0:
        raise ValueError("all_moves_by_name is empty. Check moves_pickle_path.")


def _normalize_moves_dict(moves_obj):
    """
    Internal helper:
    Converts pickle payload into the canonical format:
      all_moves_by_name[name_lower] = list_of_move_dicts
    """
    if not isinstance(moves_obj, dict) or len(moves_obj) == 0:
        raise ValueError("moves pickle must be a non empty dict.")

    # Case A: keyed by dex number (int)
    if all(isinstance(k, int) for k in moves_obj.keys()):
        if df_poke is None:
            raise RuntimeError("df_poke must be loaded before normalizing dex keyed moves.")

        id_to_name = dict(zip(df_poke["id"].astype(int), df_poke["Name"].astype(str).str.lower()))
        out = {}
        for pid, moves in moves_obj.items():
            nm = id_to_name.get(int(pid))
            if nm is not None:
                out[nm] = list(moves)
        return out

    # Case B: keyed by name (string)
    if all(isinstance(k, str) for k in moves_obj.keys()):
        out = {}
        for name, moves in moves_obj.items():
            out[str(name).strip().lower()] = list(moves)
        return out

    # Mixed keys is a sign something is off
    raise ValueError("moves pickle dict has mixed key types. Use all int dex keys OR all str name keys.")


def type_advantage(attacking_type, defending_type):
    """
    Look up type multiplier from the loaded type chart.

    Returns a float like 0.5, 1.0, 2.0, etc.
    """
    if type_chart is None:
        raise RuntimeError("Type chart not loaded. Call load_data() first.")

    a = str(attacking_type).strip().title()
    d = str(defending_type).strip().title()
    return float(type_chart.loc[a, d])


def get_pokemon_row(name):
    """
    Return a single Pokémon base stat row as a dict.
    BattleEnv expects keys like:
      'Name', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Type1', 'Type2'
    """
    if df_poke is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")

    n = str(name).strip().lower()
    hit = df_poke.loc[df_poke["Name"].astype(str).str.lower() == n]
    if len(hit) == 0:
        raise KeyError(f"Pokemon not found in df_poke: {name}")

    return hit.iloc[0].to_dict()


def get_moves_for(name):
    """
    Return the list of move dicts for a Pokémon name (lowercased lookup).
    """
    if all_moves_by_name is None:
        raise RuntimeError("Moves not loaded. Call load_data() first.")

    n = str(name).strip().lower()
    if n not in all_moves_by_name:
        raise KeyError(f"Moves not found for pokemon: {name}")
    return all_moves_by_name[n]
