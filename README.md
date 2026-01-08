# pokemon-battle-sim-ui-pygame-rl
A small Pokémon battle game built with Pygame: pick two Pokémon, battle turn by turn with 4 move choices, animated FX + scrolling combat log, and an optional PyTorch RL agent opponent (or random AI). Includes offline data loading, safe asset caching, and heavily commented code for learning and hacking.

## Pokemon Pygame

A small Pokémon battle project that connects three things into one clean pipeline:

1. **Data layer** that loads Pokémon stats, types, and moves from local files  
2. **Battle simulator / RL environment** that runs the combat rules and produces observations + rewards  
3. **Pygame front end** that turns battles into an interactive game (plus an optional trained agent opponent)

The end product is a playable “pick two Pokémon, battle, pick moves” game that can run either:
- **Human vs random opponent**, or
- **Human vs trained DQN agent** (PyTorch checkpoint)

The code is heavily commented on purpose so it can double as a reference project for:
- game state machines
- event queue based animations
- asset caching and safe loading
- RL environment design (observations, rewards, episode flow)
- plugging a trained neural net into gameplay

---

## What you can do in the game

- Search and select your Pokémon and your opponent
- Start a battle with a clean UI (HP bars, sprites, scrolling log)
- Pick one of four moves each turn
- Watch attack and status animations synced to the combat log
- Toggle opponent mode between **Random** and **Agent**
- Restart the same matchup without re selecting Pokémon (same movesets, new seed)

---

## High level architecture

This repo is structured so **logic is separated from presentation**:

- `data.py` is the only thing that knows how to read datasets  
- `env.py` is the only thing that knows battle rules  
- `model.py` is generic RL plumbing (neural net + replay buffer)  
- `train_world.py` is the training loop that learns a policy from simulated battles  
- `main.py` (Pygame UI) is “just” the front end that calls into the environment and plays back events  
- `build_moves_pickle.py` is a one time data preparation script that turns PokeAPI moves into a fast offline dataset

That separation is the whole point: you can change datasets, train differently, or rebuild the UI without rewriting the battle mechanics.

---

## File by file overview

### `data.py`
This module is the loader and single source of truth for game data. It reads the Pokémon stats table, the type effectiveness chart, and the precomputed moves dictionary, then exposes helpers like `get_pokemon_row()` and `type_advantage()` that every other file relies on. The key idea is: nothing in training or simulation hardcodes Pokémon numbers, types, or moves. Everything flows from what `data.py` loads. That makes runs reproducible, lets you swap datasets without rewriting logic, and keeps the environment clean because the env asks `data.py` questions instead of parsing CSVs itself.

### `env.py`
This file is the actual battle simulator and reinforcement learning environment. It defines the rules of combat: turn order (priority then speed), damage calculation (level term, power, attack defense ratio, STAB, type effectiveness, crits, random factor), and status logic (permanent vs temporary effects, durations, skip turn checks, end of round damage). It also builds the observation vector that the agent learns from: normalized HP, normalized stats, status one hots, type one hots, plus a feature block for each of the player’s four moves and a public block for the opponent’s moves that starts unknown until seen. In other words: `env.py` is the world, the physics, and the sensor model in one, returning `(obs, reward, done, info)` like a standard RL environment.

### `model.py`
This module contains the learning building blocks: the Q network and the replay buffer. `QNet` is a small multilayer perceptron that maps a single observation vector to four Q values, one per move slot, so action selection becomes “pick the move with the highest predicted value.” `ReplayBuffer` stores transitions `(state, action, reward, next_state, done)` and samples random minibatches, which is the classic DQN trick to reduce correlation and stabilize training. Nothing here knows anything about Pokémon. It is generic RL plumbing: a function approximator plus memory for off policy learning.

### `train_world.py`
This script is the training loop that turns the simulator into a learning system. It loads the datasets using `data.py`, repeatedly samples random Pokémon matchups from the full pool, and runs episodes in `BattleEnv` while choosing actions with an epsilon greedy policy (random moves early, increasingly greedy later). Each step is stored into the replay buffer, and after a warmup period the script performs DQN updates: predict `Q(s,a)` with the online network, build a bootstrapped target using the target network `max_a' Q_target(s2,a')`, and minimize a Huber loss between them. The target network is synced periodically to reduce moving target instability. The script also logs progress (episode, global steps, suite win rate, time) to a JSONL file and saves checkpoints that include model weights, optimizer state, and metadata so training can resume exactly where it left off.

### `build_moves_pickle.py`
This is a one time data preparation utility that downloads Pokémon move data from PokeAPI and converts it into a compact local dataset for offline training. For each Pokémon ID in a configurable range, it fetches the Pokémon JSON, follows each move URL to retrieve move metadata, and normalizes each damaging move into a small dictionary containing the move name, type, damage class, power, accuracy, PP, and priority. Status moves are intentionally filtered out so the resulting dataset matches the environment’s assumption that each move is directly damage dealing. To avoid repeatedly downloading the same move details for many Pokémon, the script uses an in memory cache keyed by the move URL, which dramatically reduces network calls. The final result is saved as a pickle so the training and UI can load all moves instantly without touching the network.

### `main.py` (Pygame UI)
This file is the playable front end. It sets up Pygame, loads audio and visuals, and runs a small state machine with three screens: `select`, `battle`, and `end`. The selection screen lets you search Pokémon names, scroll lists, and pick both sides. Icons are loaded from disk when possible and otherwise fetched from the PokeAPI in background threads to keep the UI responsive.

When a battle starts, `main.py` calls into `BattleEnv` and uses the environment’s detailed transcript output to build a queue of timed UI events: attack starts, status sounds, text lines, HP changes, and faint events. The battle screen then consumes that event queue with delays and typewriter style text, while also running attack animations and status overlays. This event queue design is what keeps combat readable and game like: instead of printing everything instantly, it plays back the turn in a paced, animated sequence. Finally, the end screen freezes the last state, shows the full log, and lets you replay the same matchup or exit back to selection.

If agent mode is enabled, `main.py` loads a PyTorch checkpoint once and uses the Q network to select the opponent’s action from the current observation. The UI does not “know” battle logic or damage math. It just asks the environment to step and then visualizes what happened.

---

## Folders and assets

- `sprites/`  
  Cached Pokémon icons used in the selection UI

- `sprites_front/` and `sprites_back/`  
  Battle sprites for opponent and player sides

- `special_effects/`  
  PNG layers for attack and status animations

- `sound_effects/`  
  Background music and move/status sound effects

- `world_runs/`  
  Training logs and saved checkpoints for the RL agent

- Dataset inputs (expected in repo root or configured paths)  
  - `pokemon_kanto_johto_sinnoh.csv`  
  - `type_chart.csv`  
  - `all_moves_by_name.pkl`

---

## Design choices worth knowing

- Battle mechanics live in `env.py`, not in the UI  
- The UI plays combat via an event queue, not by dumping text instantly  
- Missing icons/sprites/sounds fall back to placeholders instead of crashing  
- The agent code is optional: the project still works without a checkpoint

---

## Disclaimer

This is not trying to be a full Pokémon clone. It is a focused playable demo that’s built to be readable, modifiable, and good for learning: a battle simulator, an RL agent training loop, and a Pygame game layer glued together cleanly.

---

## Author
Pablo Caparrós Calle  
GitHub: @PCC23  


