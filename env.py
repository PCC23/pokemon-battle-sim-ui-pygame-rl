# env.py
# Battle environment plus all combat mechanics used by BattleEnv
# This file depends on data.py being loaded once at program start

import random  # random rolls for accuracy, crits, status procs, and move selection
import operator as op  # arithmetic helpers, used throughout to keep operations explicit

import numpy as np  # numeric arrays used for observation vectors
import pandas as pd  # used mainly to safely detect missing Type2 values

import data  # project module that provides pokemon rows, move lists, and type effectiveness

dash = chr(45)  # character used as a separator in some move names, later replaced with a space


def clean_text(x):
    # Convert any value to string and replace the dash character with a space for readable logs
    return str(x).replace(dash, " ")


def _has_type2(p):
    # Return True if the pokemon row has a valid second type value
    t2 = p.get("Type2")
    return t2 is not None and not (isinstance(t2, float) and pd.isna(t2))


def _has_stab(attacker, move_type):
    # Check Same Type Attack Bonus: move type matches Type1 or Type2
    t1 = str(attacker["Type1"]).strip().lower()
    if move_type == t1:
        return True
    if _has_type2(attacker):
        t2 = str(attacker["Type2"]).strip().lower()
        return move_type == t2
    return False


def init_status():
    # Initialize status container for a single pokemon
    # permanent is either None or {"name": "..."}
    # temporary is either None or {"name": "...", "turns_left": int}
    # flinch is a one turn flag tracked separately for convenience
    return {"permanent": None, "temporary": None, "flinch": False}


def get_turns_left(st):
    # Helper to read the remaining duration for temporary status
    t = (st or {}).get("temporary")
    if isinstance(t, dict):
        return t.get("turns_left")
    return None


def get_perm_name(st):
    # Helper to read the permanent status name, regardless of storage format
    p = (st or {}).get("permanent")
    if p is None:
        return None
    if isinstance(p, str):
        return p
    return p.get("name")


def get_temp_name(st):
    # Helper to read the temporary status name, regardless of storage format
    t = (st or {}).get("temporary")
    if t is None:
        return None
    if isinstance(t, str):
        return t
    return t.get("name")


def speed_multiplier(st):
    # Compute the speed multiplier caused by active statuses
    perm = get_perm_name(st)
    temp = get_temp_name(st)

    mult = 1.0  # baseline speed multiplier
    if perm == "paralysis":
        mult = op.mul(mult, 0.5)  # paralysis slows the pokemon
    if temp == "muddied":
        mult = op.mul(mult, 0.75)  # muddied slows the pokemon
    return float(mult)


def can_apply_permanent(st, new_name):
    # Permanent statuses only apply if there is no current permanent status
    if (st or {}).get("permanent") is not None:
        return False
    # paralysis and muddied exclusion
    if new_name == "paralysis" and get_temp_name(st) == "muddied":
        return False
    return True


def can_apply_temporary(st, new_name):
    # Temporary statuses only apply if there is no current temporary status
    if (st or {}).get("temporary") is not None:
        return False
    # paralysis and muddied exclusion
    if new_name == "muddied" and get_perm_name(st) == "paralysis":
        return False
    return True


def apply_permanent(st, name):
    # Apply a permanent status if allowed
    if not can_apply_permanent(st, name):
        return False
    st["permanent"] = {"name": name}
    return True


def apply_temporary(st, name, turns_left):
    # Apply a temporary status with duration if allowed
    if not can_apply_temporary(st, name):
        return False
    st["temporary"] = {"name": name, "turns_left": int(turns_left)}
    return True


def decrement_temporary(st):
    # Reduce remaining turns for the current temporary status, clearing it at zero
    t = st.get("temporary")
    if not isinstance(t, dict):
        return 0

    tleft = int(t.get("turns_left", 0))  # read remaining turns, default to zero
    tleft = int(op.sub(tleft, 1))  # subtract one turn
    t["turns_left"] = tleft  # write updated turns back

    if tleft <= 0:
        st["temporary"] = None  # clear temporary status when it expires
        return 0

    return tleft  # return remaining turns after decrement


def end_of_round_dot(pokemon_row, current_hp, st, debug=False):
    # Apply end of round damage over time for burn and poisoned
    max_hp = float(pokemon_row["HP"])
    perm = get_perm_name(st)

    dmg = 0.0  # default no damage
    if perm == "burn":
        dmg = op.mul(0.08, max_hp)  # burn chip damage
    elif perm == "poisoned":
        dmg = op.mul(0.08, max_hp)  # poison chip damage

    if dmg > 0.0:
        new_hp = max(0.0, op.sub(float(current_hp), float(dmg)))  # reduce HP but do not go below zero
        if debug:
            print(
                pokemon_row["Name"],
                "takes damage from",
                perm,
                round(float(dmg), 2),
                "HP",
                round(float(current_hp), 2),
                "to",
                round(float(new_hp), 2),
            )
        return float(new_hp)

    return float(current_hp)  # no change if no damage over time applies


def should_skip_action(pokemon_row, current_hp, st, debug=False):
    # Decide whether the pokemon can act this turn and update HP for self damage cases
    name = pokemon_row["Name"]

    # flinch is a one turn flag
    if st.get("flinch"):
        if debug:
            print(name, "flinched and cannot act this turn")
        st["flinch"] = False  # clear flinch after it blocks one action
        return False, float(current_hp)

    temp = get_temp_name(st)  # current temporary status name
    perm = get_perm_name(st)  # current permanent status name

    if temp == "sleep":
        # Sleep blocks action, then duration ticks down
        if debug:
            print(name, "is asleep", "turns_left:", get_turns_left(st))
        decrement_temporary(st)
        return False, float(current_hp)

    if temp == "freeze":
        # Freeze blocks action, then duration ticks down
        if debug:
            print(name, "is frozen", "turns_left:", get_turns_left(st))
        decrement_temporary(st)
        return False, float(current_hp)

    if temp == "muddied":
        # Muddied has a chance to block action each turn
        roll = random.random()
        if roll < 0.50:
            if debug:
                print(name, "is muddied and cannot act", "roll:", round(roll, 3), "turns_left:", get_turns_left(st))
            decrement_temporary(st)
            return False, float(current_hp)
        else:
            if debug:
                print(name, "is muddied but still attacks", "roll:", round(roll, 3), "turns_left:", get_turns_left(st))
        decrement_temporary(st)

    if temp == "confusion":
        # Confusion has a chance to cause self damage and block action
        roll = random.random()
        if roll < 0.50:
            max_hp = float(pokemon_row["HP"])
            self_dmg = op.mul(0.08, max_hp)  # self damage as a fraction of max HP
            new_hp = max(0.0, op.sub(float(current_hp), float(self_dmg)))
            if debug:
                print(name, "hurt itself in confusion", "self_damage:", round(float(self_dmg), 2), "turns_left:", get_turns_left(st))
            decrement_temporary(st)
            return False, float(new_hp)
        else:
            if debug:
                print(name, "is confused but still attacks", "roll:", round(roll, 3), "turns_left:", get_turns_left(st))
        decrement_temporary(st)

    if perm == "paralysis":
        # Paralysis has a chance to block action each turn
        roll = random.random()
        if roll < 0.50:
            if debug:
                print(name, "is paralyzed and cannot act", "roll:", round(roll, 3))
            return False, float(current_hp)
        else:
            if debug:
                print(name, "is paralyzed but still attacks", "roll:", round(roll, 3))

    # If nothing blocked the move, the pokemon can act and HP is unchanged
    return True, float(current_hp)


def apply_stat_mods_for_damage(pokemon_row, st, move, is_attacker):
    # Returns a copy of pokemon_row with modified stats for damage calc only
    # burn and poison reduce physical Attack
    # shattered_armour reduces physical Defense

    out = dict(pokemon_row)  # copy, so we do not mutate the original pokemon data
    perm = get_perm_name(st)  # active permanent status name
    dmg_class = str(move.get("damage_class", "")).strip().lower()  # physical or special

    if is_attacker and dmg_class == "physical":
        # Apply attacker debuffs to physical Attack
        if perm == "burn":
            out["Attack"] = op.mul(float(out["Attack"]), 0.5)
        elif perm == "poisoned":
            out["Attack"] = op.mul(float(out["Attack"]), 0.75)

    if (not is_attacker) and dmg_class == "physical":
        # Apply defender debuffs to physical Defense
        if perm == "shattered_armour":
            out["Defense"] = op.mul(float(out["Defense"]), 0.75)

    return out  # modified row used only inside damage calculation


def calculate_damage(attacker, defender, move, battle_level, crit_chance=0.10, crit_mult=1.5, debug=True):
    # Core damage formula with accuracy, STAB, type effectiveness, crit, and random factor

    def log(*args, **kwargs):
        # Local logger that only prints when debug is enabled
        if debug:
            print(*args, **kwargs)

    move_name = move["move"]  # move name string
    move_type = str(move["type"]).strip().lower()  # move type normalized
    damage_class = move["damage_class"]  # physical or special
    power = move.get("power")  # base power, may be None
    accuracy = move.get("accuracy")  # accuracy percent, may be None

    log(f"{attacker['Name']} attacks:")

    if power is None:
        # If power missing, use a default value
        log("No damage specified: default move power is 40")
        power = 40

    if accuracy is not None:
        # Accuracy roll: if roll exceeds accuracy threshold, the move misses
        roll = random.random()
        log(f"ACCURACY CHECK: acc={accuracy} roll={roll:.3f} threshold={float(accuracy)/100.0:.3f}")
        if roll >= op.truediv(float(accuracy), 100.0):
            log(f"MISSED!!!: accuracy={accuracy}% roll={roll:.3f}")
            return 0.0

    if damage_class == "physical":
        # Physical moves use Attack versus Defense
        atk_stat = float(attacker["Attack"])
        def_stat = float(defender["Defense"])
    elif damage_class == "special":
        # Special moves use Sp. Atk versus Sp. Def
        atk_stat = float(attacker["Sp. Atk"])
        def_stat = float(defender["Sp. Def"])
    else:
        # Status moves or unknown classes do no damage here
        log("No damage: damage class is not physical or special")
        return 0.0

    # Level term from classic style damage formulas
    level_term = op.add(op.truediv(op.mul(2.0, float(battle_level)), 5.0), 2.0)

    # Attack to defense ratio influences damage
    ratio = op.truediv(atk_stat, def_stat)
    ratio_exp = 1.2  # exponent that shapes how strongly stats matter
    base_damage = op.add(op.truediv(op.mul(op.mul(level_term, float(power)), float(ratio ** ratio_exp)), 50.0), 2.0)

    # Type effectiveness against defender Type1
    eff1 = float(data.type_advantage(move_type, defender["Type1"]))
    eff = eff1

    # If defender has Type2, multiply effectiveness
    eff2 = None
    if defender.get("Type2") is not None and pd.notna(defender["Type2"]):
        eff2 = float(data.type_advantage(move_type, defender["Type2"]))
        eff = op.mul(eff1, eff2)

    # STAB bonus if move type matches attacker type
    stab = 1.0
    if move_type == str(attacker["Type1"]).strip().lower():
        stab = 1.5
    elif attacker.get("Type2") is not None and pd.notna(attacker["Type2"]):
        if move_type == str(attacker["Type2"]).strip().lower():
            stab = 1.5

    # Critical hit roll
    crit_roll = random.random()
    did_crit = crit_roll < float(crit_chance)
    crit = float(crit_mult) if did_crit else 1.0

    # Random damage factor to add variance
    rand = random.uniform(0.85, 1.15)

    if did_crit:
        log(f"Critical hit: YES | crit_roll={crit_roll:.3f} | multiplier={crit}")

    # Final damage combines base damage with all multipliers
    damage = op.mul(op.mul(op.mul(op.mul(float(base_damage), float(eff)), float(stab)), float(crit)), float(rand))
    log(f"Final damage: {damage:.2f} from {move_name}")
    log("")

    return round(float(damage), 2)


# Map from move type to a list of statuses that type can inflict
TYPE_TO_STATUSES = {
    "electric": ["paralysis"],
    "fire": ["burn"],
    "ice": ["freeze"],
    "psychic": ["confusion", "sleep"],
    "ghost": ["confusion", "flinch", "sleep"],
    "fighting": ["confusion", "flinch"],
    "water": ["confusion", "muddied", "shattered_armour"],
    "grass": ["sleep", "poisoned"],
    "normal": ["sleep", "flinch"],
    "fairy": ["sleep", "flinch"],
    "dragon": ["flinch", "shattered_armour"],
    "dark": ["flinch", "confusion"],
    "steel": ["flinch", "shattered_armour"],
    "rock": ["muddied", "shattered_armour"],
    "ground": ["muddied"],
    "poison": ["poisoned"],
    "bug": ["poisoned"],
    "flying": ["flinch"],
}

# Base proc chances for each status, used to build weighted choices and apply the final roll
PROC = {
    "sleep": 0.07,
    "freeze": 0.07,
    "paralysis": 0.13,
    "muddied": 0.15,
    "burn": 0.10,
    "poisoned": 0.14,
    "shattered_armour": 0.15,
    "confusion": 0.20,
    "flinch": 0.20,
}

# Sets used to route statuses into permanent or temporary handling code
PERMANENT = {"burn", "poisoned", "paralysis", "shattered_armour"}
TEMPORARY = {"sleep", "freeze", "confusion", "muddied", "flinch"}


def _roll_temp_duration(status_name):
    # Choose how many turns a temporary status lasts
    if status_name in {"sleep", "freeze", "confusion"}:
        return random.randint(1, 3)
    if status_name == "muddied":
        return random.randint(3, 5)
    if status_name == "flinch":
        return 1
    return 0


def apply_status_on_hit(attacker, defender, move, status_state, attacker_acted_first, stab_proc_mult=1.25, debug=True):
    # Decide if a status is inflicted by this move hit, and update defender status_state in place
    move_type = str(move.get("type")).strip().lower()  # normalize move type
    candidates = TYPE_TO_STATUSES.get(move_type, [])  # possible statuses for this type
    if not candidates:
        return None

    # Build weights from base proc rates for the candidate statuses
    weights = [float(PROC.get(s, 0.0)) for s in candidates]
    if float(sum(weights)) <= 0.0:
        return None

    # Choose one candidate status using weighted random choice
    chosen = random.choices(candidates, weights=weights, k=1)[0]

    # Flinch only makes sense if attacker acted first this round
    if chosen == "flinch" and not attacker_acted_first:
        return None

    # Compute final chance, including STAB multiplier if move has STAB
    chance = float(PROC.get(chosen, 0.0))
    if _has_stab(attacker, move_type):
        chance = float(op.mul(chance, float(stab_proc_mult)))

    # Roll to see if the status actually procs
    roll = random.random()
    if roll >= chance:
        return None

    # Get the mutable status dict for the defender
    st = status_state[defender["Name"]]

    if chosen == "flinch":
        # Flinch is stored as a flag that blocks the next action
        st["flinch"] = True
        if debug:
            print("STATUS PROC:", defender["Name"], "got flinch", "chance=", round(chance, 3), "roll=", round(roll, 3))
        return "flinch"

    if chosen in PERMANENT:
        # Try to apply a permanent status, respecting exclusion rules
        ok = apply_permanent(st, chosen)
        if ok:
            if debug:
                print("STATUS PROC:", defender["Name"], "got", chosen, "permanent", "chance=", round(chance, 3), "roll=", round(roll, 3))
            return chosen
        return None

    if chosen in TEMPORARY:
        # Roll duration, then apply temporary status if allowed
        turns = int(_roll_temp_duration(chosen))
        ok = apply_temporary(st, chosen, turns_left=turns)
        if ok:
            if debug:
                print(
                    "STATUS PROC:",
                    defender["Name"],
                    "got",
                    chosen,
                    "temporary",
                    "turns_left=",
                    turns,
                    "chance=",
                    round(chance, 3),
                    "roll=",
                    round(roll, 3),
                )
            return chosen
        return None

    return None


def pick_four_moves(moves_list, type1, type2=None, rng=None):
    # Always returns exactly 4 moves.
    # Prefers up to one move of each of the pokemon types, then fills remaining.
    # If not enough unique damaging moves exist, it pads by repeating.
    rng = rng or random  # allow deterministic selection by passing a seeded rng

    # Normalize types to lowercase strings, treating missing Type2 as None
    t1 = str(type1).strip().lower() if type1 is not None else None
    t2 = None if type2 is None or (isinstance(type2, float) and pd.isna(type2)) else str(type2).strip().lower()

    # Keep only moves that actually deal damage
    moves = [m for m in list(moves_list) if m.get("damage_class") in ["physical", "special"]]

    # Fallback move used if there are no damaging moves available
    fallback = {
        "move": "struggle",
        "type": "normal",
        "damage_class": "physical",
        "power": 50,
        "accuracy": 100,
        "priority": 0,
    }

    # If no damaging moves exist, return four copies of the fallback move
    if len(moves) == 0:
        return [dict(fallback), dict(fallback), dict(fallback), dict(fallback)]

    def moves_of_type(t):
        # Helper: filter damaging moves by type
        return [m for m in moves if str(m.get("type")).strip().lower() == t]

    chosen = []  # selected moves
    used_names = set()  # used move names to avoid duplicates when possible

    def pick_one(candidates):
        # Choose one move at random from candidates, avoiding names already used
        candidates = [m for m in candidates if m.get("move") not in used_names]
        if not candidates:
            return None
        m = rng.choice(candidates)
        chosen.append(m)
        used_names.add(m.get("move"))
        return m

    # Prefer one move matching Type1 and one move matching Type2 if available
    if t1:
        pick_one(moves_of_type(t1))
    if t2:
        pick_one(moves_of_type(t2))

    # Fill remaining slots with other unused damaging moves
    remaining = [m for m in moves if m.get("move") not in used_names]
    need = int(op.sub(4, len(chosen)))

    if need > 0:
        if len(remaining) >= need:
            chosen.extend(rng.sample(remaining, need))
        else:
            chosen.extend(remaining)

    # If still short, pad by repeating random damaging moves
    while len(chosen) < 4:
        chosen.append(rng.choice(moves))

    return chosen[:4]  # ensure exactly four moves are returned


def pretty_moveset(ms):
    # Convert a moveset into a compact tuple list for printing or debugging
    out = []
    for m in list(ms):
        out.append((m.get("move"), m.get("type"), m.get("damage_class"), m.get("power"), m.get("accuracy"), m.get("priority")))
    return out


def make_fixed_movesets(p1_name, p2_name, seed=23):
    # Create deterministic movesets for two named pokemon using a local seeded RNG
    rng = random.Random(int(seed))

    p1 = data.get_pokemon_row(p1_name)  # fetch pokemon row for player 1
    p2 = data.get_pokemon_row(p2_name)  # fetch pokemon row for player 2

    p1_key = str(p1_name).strip().lower()  # normalized lookup key
    p2_key = str(p2_name).strip().lower()  # normalized lookup key

    # Pick four moves for each pokemon using the seeded RNG
    p1_moveset = pick_four_moves(data.all_moves_by_name[p1_key], p1["Type1"], p1.get("Type2"), rng=rng)
    p2_moveset = pick_four_moves(data.all_moves_by_name[p2_key], p2["Type1"], p2.get("Type2"), rng=rng)

    return p1_moveset, p2_moveset


FIXED_MOVESETS = {}  # placeholder dict for storing cached fixed movesets if desired


# Canonical type list used for one hot encoding in observation features
TYPE_LIST = [
    "normal",
    "fire",
    "water",
    "electric",
    "grass",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
    "fairy",
]
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPE_LIST)}  # map type string to index
TYPE_UNKNOWN = len(TYPE_LIST)  # index used when type is missing or not recognized

# Encodings for permanent status and temporary status
PERM_LIST = ["none", "burn", "poisoned", "paralysis", "shattered_armour"]
PERM_TO_IDX = {s: i for i, s in enumerate(PERM_LIST)}

TEMP_LIST = ["none", "sleep", "freeze", "confusion", "muddied"]
TEMP_TO_IDX = {s: i for i, s in enumerate(TEMP_LIST)}


def _type_idx(x):
    # Convert a type value into its integer index for one hot encoding
    if x is None:
        return TYPE_UNKNOWN
    if isinstance(x, float) and np.isnan(x):
        return TYPE_UNKNOWN
    s = str(x).strip().lower()
    return TYPE_TO_IDX.get(s, TYPE_UNKNOWN)


def _one_hot(idx, size):
    # Create a one hot vector of length size with a 1 at idx if idx is valid
    v = np.zeros(int(size), dtype=np.float32)
    j = int(idx)
    if 0 <= j < int(size):
        v[j] = 1.0
    return v


def _perm_name(st):
    # Return permanent status name for encoding, defaulting to "none"
    p = (st or {}).get("permanent")
    if p is None:
        return "none"
    if isinstance(p, str):
        return p
    return str(p.get("name", "none"))


def _temp_name(st):
    # Return temporary status name for encoding, defaulting to "none"
    t = (st or {}).get("temporary")
    if t is None:
        return "none"
    if isinstance(t, str):
        return t
    return str(t.get("name", "none"))


def _temp_turns_left(st):
    # Return remaining turns for temporary status as a float
    t = (st or {}).get("temporary")
    if isinstance(t, dict):
        return float(t.get("turns_left", 0))
    return 0.0


def _damage_class_flags(move):
    # Encode whether the move is physical or special as two numeric flags
    dc = str(move.get("damage_class", "")).strip().lower()
    is_phys = 1.0 if dc == "physical" else 0.0
    is_spec = 1.0 if dc == "special" else 0.0
    return is_phys, is_spec


def _move_features(attacker_row, defender_row, move):
    # Convert a move into a numeric feature vector for the observation
    power = move.get("power")
    if power is None:
        power = 40  # default power for missing values
    acc = move.get("accuracy")
    if acc is None:
        acc = 100  # treat missing accuracy as guaranteed hit
    pr = move.get("priority", 0)  # move priority, default zero

    move_type = str(move.get("type", "")).strip().lower()
    t_idx = TYPE_TO_IDX.get(move_type, TYPE_UNKNOWN)  # type index for one hot encoding

    stab = 1.0 if _has_stab(attacker_row, move_type) else 0.0  # STAB flag
    is_phys, is_spec = _damage_class_flags(move)  # damage class flags

    # Normalize numeric fields to bounded ranges for learning stability
    power_n = op.truediv(float(power), 200.0)
    acc_n = op.truediv(float(acc), 100.0)
    pr_n = op.truediv(op.add(float(pr), 6.0), 12.0)

    # Concatenate scalar features with a one hot type vector
    return np.concatenate(
        [
            np.array([power_n, acc_n, pr_n, stab, is_phys, is_spec], dtype=np.float32),
            _one_hot(t_idx, op.add(TYPE_UNKNOWN, 1)),
        ],
        axis=0,
    )


def _stat_pack(p):
    # Pack pokemon base stats into a normalized numeric vector
    return np.array(
        [
            op.truediv(float(p["HP"]), 255.0),
            op.truediv(float(p["Attack"]), 200.0),
            op.truediv(float(p["Defense"]), 200.0),
            op.truediv(float(p["Sp. Atk"]), 200.0),
            op.truediv(float(p["Sp. Def"]), 200.0),
            op.truediv(float(p["Speed"]), 200.0),
        ],
        dtype=np.float32,
    )


def _status_pack(st):
    # Pack status info into a numeric vector: one hot perm, one hot temp, normalized turns, flinch flag
    perm = _perm_name(st)
    temp = _temp_name(st)

    perm_idx = PERM_TO_IDX.get(perm, 0)
    temp_idx = TEMP_TO_IDX.get(temp, 0)

    tleft = _temp_turns_left(st)
    tleft_n = op.truediv(min(tleft, 5.0), 5.0)  # clamp then normalize

    fl = 1.0 if bool((st or {}).get("flinch", False)) else 0.0  # flinch flag

    return np.concatenate(
        [
            _one_hot(perm_idx, len(PERM_LIST)),
            _one_hot(temp_idx, len(TEMP_LIST)),
            np.array([tleft_n, fl], dtype=np.float32),
        ],
        axis=0,
    )


class BattleEnv:
    # Environment that simulates a one versus one battle and produces observation vectors for RL
    def __init__(self, battle_level=5, max_rounds=80, crit_chance=0.10, crit_mult=1.5):
        self.battle_level = int(battle_level)  # battle level used in damage formula
        self.max_rounds = int(max_rounds)  # hard cap on episode length
        self.crit_chance = float(crit_chance)  # probability of a crit
        self.crit_mult = float(crit_mult)  # crit damage multiplier

        self.opponent_chaos = 0.0  # reserved knob for opponent randomness, if used externally

        # Current battle participants and state
        self.p1 = None
        self.p2 = None
        self.p1_hp = None
        self.p2_hp = None
        self.p1_moveset = None
        self.p2_moveset = None
        self.status_state = None
        self.round_num = None

        # Tracking which opponent moves have been revealed to the agent
        self.p2_seen = None
        self.p2_seen_feats = None
        self._p2_unknown_vec = None

    def reset(self, p1_name, p2_name, p1_moveset=None, p2_moveset=None, opponent_chaos=None, seed=None):
        # Reset environment to a fresh battle and return the initial observation
        if seed is not None:
            random.seed(int(seed))  # seed python random for reproducibility
            np.random.seed(int(seed))  # seed numpy random for reproducibility

        if opponent_chaos is not None:
            self.opponent_chaos = float(opponent_chaos)  # store external chaos setting

        # Load pokemon rows for both sides
        self.p1 = data.get_pokemon_row(p1_name)
        self.p2 = data.get_pokemon_row(p2_name)

        # Normalize names for lookup into move dictionaries
        p1_key = str(p1_name).strip().lower()
        p2_key = str(p2_name).strip().lower()

        # Choose movesets unless provided externally
        if p1_moveset is None:
            self.p1_moveset = pick_four_moves(data.all_moves_by_name[p1_key], self.p1["Type1"], self.p1.get("Type2"))
        else:
            self.p1_moveset = list(p1_moveset)

        if p2_moveset is None:
            self.p2_moveset = pick_four_moves(data.all_moves_by_name[p2_key], self.p2["Type1"], self.p2.get("Type2"))
        else:
            self.p2_moveset = list(p2_moveset)

        # Initialize HP at max values
        self.p1_hp = float(self.p1["HP"])
        self.p2_hp = float(self.p2["HP"])

        # Initialize status states for both pokemon
        self.status_state = {self.p1["Name"]: init_status(), self.p2["Name"]: init_status()}
        self.round_num = 1  # start at round one

        # Opponent move reveal tracking for partial observability
        self.p2_seen = [False, False, False, False]
        self.p2_seen_feats = [None, None, None, None]
        feat_len = len(_move_features(self.p2, self.p1, self.p2_moveset[0]))  # feature length for one move
        self._p2_unknown_vec = np.zeros(int(feat_len), dtype=np.float32)  # placeholder for unknown opponent move

        return self._obs()  # return initial observation vector

    def _p2_public_move_block(self):
        # Build the opponent move feature block with unknown placeholders for unseen moves
        blocks = []
        for i in range(4):
            if self.p2_seen[i] and (self.p2_seen_feats[i] is not None):
                blocks.append(self.p2_seen_feats[i])
            else:
                blocks.append(self._p2_unknown_vec)
        return np.concatenate(blocks, axis=0)

    def _obs(self):
        # Construct the full observation vector for the agent
        st1 = self.status_state[self.p1["Name"]]  # status dict for player 1
        st2 = self.status_state[self.p2["Name"]]  # status dict for player 2

        # HP as fractions of max HP
        hp1_frac = op.truediv(float(self.p1_hp), float(self.p1["HP"]))
        hp2_frac = op.truediv(float(self.p2_hp), float(self.p2["HP"]))

        # One hot encode both types for both pokemon
        types = np.concatenate(
            [
                _one_hot(_type_idx(self.p1["Type1"]), op.add(TYPE_UNKNOWN, 1)),
                _one_hot(_type_idx(self.p1.get("Type2")), op.add(TYPE_UNKNOWN, 1)),
                _one_hot(_type_idx(self.p2["Type1"]), op.add(TYPE_UNKNOWN, 1)),
                _one_hot(_type_idx(self.p2.get("Type2")), op.add(TYPE_UNKNOWN, 1)),
            ],
            axis=0,
        )

        # Core features include HP, stats, statuses, and types
        core = np.concatenate(
            [
                np.array([hp1_frac, hp2_frac], dtype=np.float32),
                _stat_pack(self.p1),
                _stat_pack(self.p2),
                _status_pack(st1),
                _status_pack(st2),
                types,
            ],
            axis=0,
        )

        # Player 1 move features are fully known
        p1_move_block = np.concatenate([_move_features(self.p1, self.p2, mv) for mv in self.p1_moveset], axis=0)

        # Opponent move features are partially observed
        p2_public_block = self._p2_public_move_block()

        # Final observation is a concatenation of all blocks
        return np.concatenate([core, p1_move_block, p2_public_block], axis=0).astype(np.float32)

    def _pick_turn_order(self, m1, m2):
        # Decide who goes first based on priority then effective speed, with random tie break
        pr1 = float(m1.get("priority", 0))
        pr2 = float(m2.get("priority", 0))
        if pr1 > pr2:
            return 1
        if pr2 > pr1:
            return 2

        s1 = op.mul(float(self.p1["Speed"]), float(speed_multiplier(self.status_state[self.p1["Name"]])))
        s2 = op.mul(float(self.p2["Speed"]), float(speed_multiplier(self.status_state[self.p2["Name"]])))
        if s1 > s2:
            return 1
        if s2 > s1:
            return 2

        return 1 if random.random() < 0.5 else 2  # random tie break

    def _opponent_pick(self):
        # Baseline opponent policy: choose a random move index from 0 to 3
        return int(random.randint(0, 3))

    def _fmt_type2(self, row):
        # Format Type2 for display, returning "nan" if missing
        t2 = row.get("Type2")
        if t2 is None:
            return "nan"
        if isinstance(t2, float) and pd.isna(t2):
            return "nan"
        return str(t2).strip().lower()

    def _cap_lines(self, fn):
        # Run a function while capturing any printed output lines, returning both result and captured lines
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = fn()
        lines = []
        for ln in buf.getvalue().splitlines():
            s = str(ln).rstrip()
            if len(s) > 0:
                lines.append(s)
        return out, lines

    def intro_lines(self):
        # Produce a human readable intro block describing both pokemon and their movesets
        p1 = str(self.p1["Name"]).strip().lower()
        p2 = str(self.p2["Name"]).strip().lower()

        t1a = str(self.p1["Type1"]).strip().lower()
        t1b = str(self.p2["Type1"]).strip().lower()
        t2a = self._fmt_type2(self.p1)
        t2b = self._fmt_type2(self.p2)

        out = []
        out.append("FIGHT START")
        out.append(f"{p1} Types: {t1a} {t2a} | HP: {float(self.p1['HP'])} | Speed: {float(self.p1['Speed'])}")
        out.append(f"{p2} Types: {t1b} {t2b} | HP: {float(self.p2['HP'])} | Speed: {float(self.p2['Speed'])}")
        out.append("")
        out.append(f"{p1} moveset:")
        for m in list(self.p1_moveset):
            mv = clean_text(m.get("move", "move"))
            mt = str(m.get("type", "")).strip().lower()
            pr = int(m.get("priority", 0))
            out.append(f"{mv} | {mt} | prio: {pr}")
        out.append("")
        out.append(f"{p2} moveset:")
        for m in list(self.p2_moveset):
            mv = clean_text(m.get("move", "move"))
            mt = str(m.get("type", "")).strip().lower()
            pr = int(m.get("priority", 0))
            out.append(f"{mv} | {mt} | prio: {pr}")
        out.append("")
        out.append("===========")
        return out

    def _pick_turn_order_detailed(self, m1, m2):
        # Turn order with an explanatory string for logging
        p1 = str(self.p1["Name"]).strip().lower()
        p2 = str(self.p2["Name"]).strip().lower()

        pr1 = float(m1.get("priority", 0))
        pr2 = float(m2.get("priority", 0))

        s1 = float(op.mul(float(self.p1["Speed"]), float(speed_multiplier(self.status_state[self.p1["Name"]]))))
        s2 = float(op.mul(float(self.p2["Speed"]), float(speed_multiplier(self.status_state[self.p2["Name"]]))))

        if pr1 > pr2:
            return 1, f"Turn order: {p1} goes first higher priority {pr1} vs {pr2}"
        if pr2 > pr1:
            return 2, f"Turn order: {p2} goes first higher priority {pr2} vs {pr1}"

        if s1 > s2:
            return 1, f"Turn order: {p1} goes first speed {s1} vs {s2}"
        if s2 > s1:
            return 2, f"Turn order: {p2} goes first speed {s2} vs {s1}"

        first = 1 if random.random() < 0.5 else 2
        who = p1 if first == 1 else p2
        return first, f"Turn order: tie on priority and speed, random pick → {who} goes first"

    def _do_attack_detailed(
        self,
        attacker_row,
        defender_row,
        move,
        defender_hp,
        attacker_hp,
        attacker_acted_first,
        defender_will_act_later,
    ):
        # Execute one attack action with detailed logging and status proc handling
        a_name = str(attacker_row["Name"]).strip().lower()
        d_name = str(defender_row["Name"]).strip().lower()

        a_st = self.status_state[attacker_row["Name"]]  # attacker status dict
        d_st = self.status_state[defender_row["Name"]]  # defender status dict

        lines = []  # collected log lines for this attack

        def do_skip():
            # Wrapper so we can capture printed lines from should_skip_action
            return should_skip_action(attacker_row, attacker_hp, a_st, debug=True)

        # Determine if attacker can act, capturing any debug output
        (can_act, new_attacker_hp), skip_lines = self._cap_lines(do_skip)
        attacker_hp = float(new_attacker_hp)

        for s in skip_lines:
            lines.append(s)

        # If attacker fainted due to self damage, end immediately
        if attacker_hp <= 0.0:
            return float(defender_hp), float(attacker_hp), 0.0, False, lines

        # If attacker cannot act due to status, no damage is dealt
        if not can_act:
            return float(defender_hp), float(attacker_hp), 0.0, False, lines

        # Apply status based stat modifiers for damage calculation only
        attacker_mod = apply_stat_mods_for_damage(attacker_row, a_st, move, is_attacker=True)
        defender_mod = apply_stat_mods_for_damage(defender_row, d_st, move, is_attacker=False)

        def do_dmg():
            # Wrapper so we can capture printed lines from calculate_damage
            return calculate_damage(
                attacker_mod,
                defender_mod,
                move,
                battle_level=self.battle_level,
                crit_chance=self.crit_chance,
                crit_mult=self.crit_mult,
                debug=True,
            )

        # Compute damage, capturing debug lines
        dmg, dmg_lines = self._cap_lines(do_dmg)
        dmg = float(dmg)

        for s in dmg_lines:
            lines.append(s)

        # Apply damage to defender HP
        def_hp_before = float(defender_hp)
        new_def_hp = max(0.0, op.sub(def_hp_before, dmg))

        # Log the attack summary line
        mv_name = clean_text(move.get("move", "move"))
        mv_type = str(move.get("type", "")).strip().lower()
        lines.append(
            f"ATTACK: {a_name} uses {mv_name} | move_type: {mv_type} | damage: {round(dmg, 2)}"
        )

        # Determine whether defender loses its upcoming action due to flinch
        skip_defender_turn = False
        if dmg > 0.0 and new_def_hp > 0.0:

            def do_proc():
                # Wrapper so we can capture printed lines from apply_status_on_hit
                return apply_status_on_hit(
                    attacker_row,
                    defender_row,
                    move,
                    status_state=self.status_state,
                    attacker_acted_first=attacker_acted_first,
                    debug=True,
                )

            # Apply status proc roll, capturing debug output
            inflicted, proc_lines = self._cap_lines(do_proc)

            for s in proc_lines:
                lines.append(s)

            # If defender will act later and got flinch, mark that action to be skipped
            if defender_will_act_later and inflicted == "flinch":
                skip_defender_turn = True
                lines.append("")
                lines.append(f"{d_name} flinched and loses its action this round")
                lines.append("")

        return float(new_def_hp), float(attacker_hp), float(dmg), bool(skip_defender_turn), lines

    def step_detailed(self, action_idx, opp_action_idx=None):
        # Full environment step with transcript lines and events snapshots for animation or UI
        transcript = []
        events = []

        # If battle already ended, return done immediately with empty logs
        if self.p1_hp <= 0.0 or self.p2_hp <= 0.0:
            info = {"p1_hp": float(self.p1_hp), "p2_hp": float(self.p2_hp), "round": int(self.round_num or 0)}
            return self._obs(), 0.0, True, info, transcript, events

        # Sanitize action index into the valid range 0 to 3
        a = int(action_idx)
        if a < 0 or a > 3:
            a = 0

        # Select player 1 move
        m1 = self.p1_moveset[a]

        # Select opponent move either from argument or random policy
        if opp_action_idx is None:
            opp_idx = self._opponent_pick()
        else:
            opp_idx = int(opp_action_idx)
            if opp_idx < 0 or opp_idx > 3:
                opp_idx = 0

        m2 = self.p2_moveset[int(opp_idx)]

        # Mark this opponent move as seen and store its feature vector
        self.p2_seen[int(opp_idx)] = True
        self.p2_seen_feats[int(opp_idx)] = _move_features(self.p2, self.p1, m2)

        # Labels used in logs
        p1_label = str(self.p1["Name"]).strip().lower()
        p2_label = str(self.p2["Name"]).strip().lower()

        round_now = int(self.round_num)

        # Round header and chosen moves
        transcript.append("")
        transcript.append(f"ROUND {round_now}")
        transcript.append(f"{p1_label} chose: {clean_text(m1.get('move', 'move'))} | prio: {int(m1.get('priority', 0))}")
        transcript.append(f"{p2_label} chose: {clean_text(m2.get('move', 'move'))} | prio: {int(m2.get('priority', 0))}")
        transcript.append("")

        # Snapshot HP before the round for reward computation
        p1_before = float(self.p1_hp)
        p2_before = float(self.p2_hp)

        # Decide turn order and log why
        first, order_line = self._pick_turn_order_detailed(m1, m2)
        transcript.append(order_line)
        transcript.append("")

        # Execute attacks in the decided order, respecting flinch skip flags
        if first == 1:
            self.p2_hp, self.p1_hp, _, skip_p2, lines1 = self._do_attack_detailed(self.p1, self.p2, m1, self.p2_hp, self.p1_hp, True, True)
            for s in lines1:
                transcript.append(s)
            events.append({"p1_hp": float(self.p1_hp), "p2_hp": float(self.p2_hp)})

            if self.p2_hp > 0.0 and not skip_p2:
                self.p1_hp, self.p2_hp, _, _, lines2 = self._do_attack_detailed(self.p2, self.p1, m2, self.p1_hp, self.p2_hp, False, False)
                for s in lines2:
                    transcript.append(s)
                events.append({"p1_hp": float(self.p1_hp), "p2_hp": float(self.p2_hp)})
        else:
            self.p1_hp, self.p2_hp, _, skip_p1, lines1 = self._do_attack_detailed(self.p2, self.p1, m2, self.p1_hp, self.p2_hp, True, True)
            for s in lines1:
                transcript.append(s)
            events.append({"p1_hp": float(self.p1_hp), "p2_hp": float(self.p2_hp)})

            if self.p1_hp > 0.0 and not skip_p1:
                self.p2_hp, self.p1_hp, _, _, lines2 = self._do_attack_detailed(self.p1, self.p2, m1, self.p2_hp, self.p1_hp, False, False)
                for s in lines2:
                    transcript.append(s)
                events.append({"p1_hp": float(self.p1_hp), "p2_hp": float(self.p2_hp)})

        # Apply end of round damage over time with captured debug logs
        def do_dot1():
            return end_of_round_dot(self.p1, self.p1_hp, self.status_state[self.p1["Name"]], debug=True)

        def do_dot2():
            return end_of_round_dot(self.p2, self.p2_hp, self.status_state[self.p2["Name"]], debug=True)

        new_p1, dot_lines1 = self._cap_lines(do_dot1)
        new_p2, dot_lines2 = self._cap_lines(do_dot2)
        self.p1_hp = float(new_p1)
        self.p2_hp = float(new_p2)

        events.append({"p1_hp": float(self.p1_hp), "p2_hp": float(self.p2_hp)})

        for s in dot_lines1:
            transcript.append(s)
        for s in dot_lines2:
            transcript.append(s)

        # End of round summary
        transcript.append(
            f"END ROUND {round_now} | {p1_label} HP: {round(float(self.p1_hp), 2)} | {p2_label} HP: {round(float(self.p2_hp), 2)}"
        )
        transcript.append("===========")

        # Compute dealt and taken damage during the round for reward shaping
        dealt = max(0.0, op.sub(p2_before, float(self.p2_hp)))
        taken = max(0.0, op.sub(p1_before, float(self.p1_hp)))

        dealt_n = op.truediv(dealt, float(self.p2["HP"]))
        taken_n = op.truediv(taken, float(self.p1["HP"]))

        # Reward encourages dealing damage and discourages taking damage
        reward = float(dealt_n)
        reward = float(op.add(reward, op.sub(0.0, op.mul(0.5, float(taken_n)))))

        # Determine terminal conditions and add win or loss bonus
        done = False
        if self.p2_hp <= 0.0 and self.p1_hp > 0.0:
            done = True
            reward = float(op.add(reward, 1.0))
            transcript.append("")
            transcript.append(f"{p2_label} fainted. {p1_label} wins.")
        elif self.p1_hp <= 0.0 and self.p2_hp > 0.0:
            done = True
            reward = float(op.sub(reward, 1.0))
            transcript.append("")
            transcript.append(f"{p1_label} fainted. {p2_label} wins.")
        elif int(self.round_num) >= int(self.max_rounds):
            done = True

        # Advance round counter
        self.round_num = int(op.add(int(self.round_num), 1))

        # Pack info dict returned to caller
        info = {
            "p1_hp": round(float(self.p1_hp), 2),
            "p2_hp": round(float(self.p2_hp), 2),
            "round": int(self.round_num),
            "first": int(first),
            "opp_idx": int(opp_idx),
        }

        return self._obs(), float(reward), bool(done), info, transcript, events

    def step(self, action_idx):
        # Lightweight step used for training loops, returns only obs, reward, done, info
        if self.p1_hp <= 0.0 or self.p2_hp <= 0.0:
            info = {"p1_hp": float(self.p1_hp), "p2_hp": float(self.p2_hp), "round": int(self.round_num or 0)}
            return self._obs(), 0.0, True, info

        # Sanitize action index
        a = int(action_idx)
        if a < 0 or a > 3:
            a = 0

        m1 = self.p1_moveset[a]  # chosen move for player 1

        opp_idx = self._opponent_pick()  # choose opponent move
        m2 = self.p2_moveset[int(opp_idx)]

        # Update observed opponent move features
        self.p2_seen[int(opp_idx)] = True
        self.p2_seen_feats[int(opp_idx)] = _move_features(self.p2, self.p1, m2)

        # Save HP before actions for reward computation
        p1_before = float(self.p1_hp)
        p2_before = float(self.p2_hp)

        # Decide turn order
        first = self._pick_turn_order(m1, m2)

        # Execute attacks, respecting flinch skip flags
        if first == 1:
            self.p2_hp, self.p1_hp, _, skip_p2, _ = self._do_attack_detailed(self.p1, self.p2, m1, self.p2_hp, self.p1_hp, True, True)
            if self.p2_hp > 0.0 and not skip_p2:
                self.p1_hp, self.p2_hp, _, _, _ = self._do_attack_detailed(self.p2, self.p1, m2, self.p1_hp, self.p2_hp, False, False)
        else:
            self.p1_hp, self.p2_hp, _, skip_p1, _ = self._do_attack_detailed(self.p2, self.p1, m2, self.p1_hp, self.p2_hp, True, True)
            if self.p1_hp > 0.0 and not skip_p1:
                self.p2_hp, self.p1_hp, _, _, _ = self._do_attack_detailed(self.p1, self.p2, m1, self.p2_hp, self.p1_hp, False, False)

        # Apply end of round damage over time without printing
        self.p1_hp = end_of_round_dot(self.p1, self.p1_hp, self.status_state[self.p1["Name"]], debug=False)
        self.p2_hp = end_of_round_dot(self.p2, self.p2_hp, self.status_state[self.p2["Name"]], debug=False)

        # Compute dealt and taken damage for reward shaping
        dealt = max(0.0, op.sub(p2_before, float(self.p2_hp)))
        taken = max(0.0, op.sub(p1_before, float(self.p1_hp)))

        dealt_n = op.truediv(dealt, float(self.p2["HP"]))
        taken_n = op.truediv(taken, float(self.p1["HP"]))

        reward = float(dealt_n)
        reward = float(op.add(reward, op.sub(0.0, op.mul(0.5, float(taken_n)))))

        # Determine terminal conditions and add terminal bonus
        done = False
        if self.p2_hp <= 0.0 and self.p1_hp > 0.0:
            done = True
            reward = float(op.add(reward, 1.0))
        elif self.p1_hp <= 0.0 and self.p2_hp > 0.0:
            done = True
            reward = float(op.sub(reward, 1.0))
        elif int(self.round_num) >= int(self.max_rounds):
            done = True

        # Advance round counter
        self.round_num = int(op.add(int(self.round_num), 1))

        # Return compact info dict
        info = {"p1_hp": round(float(self.p1_hp), 2), "p2_hp": round(float(self.p2_hp), 2), "round": int(self.round_num)}
        return self._obs(), float(reward), bool(done), info
