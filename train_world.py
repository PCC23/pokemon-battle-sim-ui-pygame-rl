import argparse  # CLI argument parsing for training configuration
import json  # writing progress logs as JSON lines
import random  # python RNG for epsilon action selection and matchup sampling
import time  # timing training progress for logging
import operator as op  # explicit arithmetic helpers, used for consistency
from pathlib import Path  # safe path handling across OSes

import numpy as np  # numeric utilities, used here for epsilon decay math
import torch  # PyTorch core
import torch.nn.functional as F  # loss functions and functional ops
import torch.optim as optim  # optimizers

import data  # project module that loads pokemon, moves, and type chart
from env import BattleEnv  # environment that simulates battles and returns observations
from model import QNet, ReplayBuffer  # Q network and replay buffer used by DQN


def set_global_seeds(seed: int) -> None:
    # Set seeds for python, numpy, and torch so runs are repeatable
    seed_i = int(seed)  # ensure seed is an int
    random.seed(seed_i)  # seed python random
    np.random.seed(seed_i)  # seed numpy random
    torch.manual_seed(seed_i)  # seed torch CPU RNG
    if torch.cuda.is_available():  # if using GPU, also seed CUDA RNGs
        torch.cuda.manual_seed_all(seed_i)


def sample_matchup(name_pool, rng: random.Random):
    # Randomly sample two distinct pokemon names from the pool
    p1 = rng.choice(name_pool)  # pick player 1 pokemon
    p2 = rng.choice(name_pool)  # pick player 2 pokemon
    while p2 == p1:  # ensure they are not the same pokemon
        p2 = rng.choice(name_pool)
    return p1, p2  # return the matchup pair


def eps_by_step(t: int, eps_start: float, eps_end: float, eps_decay_steps: int) -> float:
    # Exponential epsilon schedule: starts high and decays toward eps_end as global steps increase
    x = op.truediv(float(t), float(eps_decay_steps))  # normalize step count to decay scale
    decay = float(np.exp(op.sub(0.0, float(x))))  # exp negative x gives a smooth decay factor
    return float(eps_end) + (float(eps_start) - float(eps_end)) * float(decay)  # interpolate toward eps_end


def save_checkpoint(path, q, opt, meta: dict):
    # Save model weights, optimizer state, and metadata into a single checkpoint file
    payload = {
        "q_state": q.state_dict(),  # Q network parameters
        "opt_state": opt.state_dict(),  # optimizer parameters and momentum buffers
        "meta": dict(meta),  # training metadata like step counters
    }
    p = Path(path)  # convert to Path object
    p.parent.mkdir(parents=True, exist_ok=True)  # ensure checkpoint folder exists
    torch.save(payload, str(p))  # serialize with torch.save


def load_checkpoint(path, q, opt, device):
    # Load checkpoint into an existing q network and optimizer
    payload = torch.load(path, map_location=device)  # load checkpoint onto current device
    q.load_state_dict(payload["q_state"])  # restore model weights
    opt.load_state_dict(payload["opt_state"])  # restore optimizer state
    meta = payload.get("meta", {})  # read metadata if present
    return q, opt, meta  # return restored objects and metadata


@torch.no_grad()
def eval_suite(q, device, name_pool, suite_pairs, n_eval_per_pair=50, seed=10000, max_steps=200):
    # Evaluate current policy greedily on a fixed suite of matchups and return win rate
    q.eval()  # switch Q network to eval mode
    env = BattleEnv()  # create a fresh environment
    wins = 0  # win counter
    total = 0  # total games evaluated

    rng = random.Random(int(seed))  # deterministic RNG for evaluation resets

    for p1, p2 in suite_pairs:  # iterate over matchup pairs
        for k in range(int(n_eval_per_pair)):  # repeat each matchup multiple times
            s = env.reset(p1, p2, p1_moveset=None, p2_moveset=None, seed=rng.randint(0, 10**9))  # reset with a fresh seed
            done = False  # episode termination flag
            steps = 0  # step counter within this episode
            while (not done) and steps < int(max_steps):  # run until terminal or max steps
                st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)  # convert obs to tensor with batch dim
                a = int(torch.argmax(q(st), dim=1).item())  # greedy action: argmax over the 4 Q values
                s, r, done, info = env.step(a)  # apply action in environment
                steps += 1  # increment episode step counter

            total += 1  # one evaluation episode completed
            if float(env.p2_hp) <= 0.0 and float(env.p1_hp) > 0.0:  # define win condition by remaining HP
                wins += 1  # increment win count

    if total <= 0:
        return 0.0  # guard against division by zero
    return float(wins) / float(total)  # win rate across the suite


def main():
    # Main training entry point: parses args, loads data, trains a DQN agent, logs and checkpoints
    ap = argparse.ArgumentParser()  # create CLI parser

    ap.add_argument("--episodes", type=int, default=5000000)  # total training episodes
    ap.add_argument("--seed", type=int, default=23)  # global seed for reproducibility

    ap.add_argument("--pokemon_csv", type=str, default="pokemon_kanto_johto_sinnoh.csv")  # pokemon stats dataset
    ap.add_argument("--type_csv", type=str, default="type_chart.csv")  # type effectiveness dataset
    ap.add_argument("--moves_pkl", type=str, default="all_moves_by_name.pkl")  # precomputed moves per pokemon

    ap.add_argument("--lr", type=float, default=3e-4)  # Adam learning rate
    ap.add_argument("--gamma", type=float, default=0.99)  # discount factor for bootstrapping

    ap.add_argument("--batch_size", type=int, default=128)  # minibatch size for replay updates
    ap.add_argument("--warmup_steps", type=int, default=5000)  # steps before starting gradient updates
    ap.add_argument("--updates_per_step", type=int, default=2)  # how many gradient steps per env step
    ap.add_argument("--target_update_every", type=int, default=500)  # frequency to sync target network

    ap.add_argument("--eps_start", type=float, default=1.0)  # starting epsilon for exploration
    ap.add_argument("--eps_end", type=float, default=0.05)  # final epsilon floor
    ap.add_argument("--eps_decay_steps", type=int, default=200000)  # decay timescale in steps

    ap.add_argument("--max_steps_per_episode", type=int, default=200)  # episode truncation limit

    ap.add_argument("--save_every", type=int, default=5000)  # checkpoint every N episodes
    ap.add_argument("--eval_every", type=int, default=5000)  # evaluate every N episodes

    ap.add_argument("--suite_size", type=int, default=64)  # number of matchup pairs in evaluation suite
    ap.add_argument("--suite_eval_per_pair", type=int, default=30)  # episodes per matchup during eval

    ap.add_argument("--out_dir", type=str, default="world_runs")  # output directory for logs and checkpoints
    ap.add_argument("--resume", type=str, default=None)  # path to checkpoint to resume from

    args = ap.parse_args()  # parse CLI args

    set_global_seeds(int(args.seed))  # seed all RNGs

    data.load_data(
        pokemon_csv_path=args.pokemon_csv,  # load pokemon stats
        type_chart_csv_path=args.type_csv,  # load type effectiveness
        moves_pickle_path=args.moves_pkl,  # load moveset dictionary
    )

    if data.df_poke is None or len(data.df_poke) == 0:
        raise RuntimeError("df_poke not loaded")  # fail fast if pokemon data missing

    name_pool = [str(x).strip().lower() for x in list(data.df_poke["Name"].astype(str).values)]  # list of usable pokemon names
    if len(name_pool) < 2:
        raise RuntimeError("name_pool too small")  # need at least two unique pokemon

    out_dir = Path(args.out_dir)  # output directory as Path
    out_dir.mkdir(parents=True, exist_ok=True)  # create output directory if needed
    log_path = out_dir / "progress.jsonl"  # training log file path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pick GPU if available

    env = BattleEnv()  # create environment instance for training
    s0 = env.reset(name_pool[0], name_pool[1], p1_moveset=None, p2_moveset=None, seed=int(args.seed))  # reset once to get obs size
    obs_dim = int(s0.shape[0])  # observation vector length

    q = QNet(obs_dim, 4).to(device)  # online Q network
    qt = QNet(obs_dim, 4).to(device)  # target Q network
    qt.load_state_dict(q.state_dict())  # initialize target equal to online
    qt.eval()  # target used only for inference during target computation

    opt = optim.Adam(q.parameters(), lr=float(args.lr))  # optimizer for online Q network
    buf = ReplayBuffer(capacity=200000)  # replay buffer storing transitions

    global_step = 0  # total environment steps taken across all episodes
    start_ep = 0  # starting episode index, used when resuming

    if args.resume:
        q, opt, meta = load_checkpoint(args.resume, q, opt, device)  # restore online network and optimizer
        global_step = int(meta.get("global_step", 0))  # restore global step
        start_ep = int(meta.get("episode", 0))  # restore episode counter
        qt.load_state_dict(q.state_dict())  # sync target with restored online net
        qt.eval()  # keep target in eval mode

    rng = random.Random(int(args.seed) + 999)  # local RNG for matchup sampling and env seeds

    suite_pairs = []  # pre sampled evaluation matchups
    for _ in range(int(args.suite_size)):
        suite_pairs.append(sample_matchup(name_pool, rng))  # build fixed suite for consistent comparisons

    start_time = time.time()  # start clock for elapsed time logging

    for ep in range(int(start_ep), int(args.episodes)):  # training loop over episodes
        p1, p2 = sample_matchup(name_pool, rng)  # sample random matchup

        s = env.reset(p1, p2, p1_moveset=None, p2_moveset=None, seed=rng.randint(0, 10**9))  # reset env for this episode

        done = False  # episode done flag
        steps = 0  # steps within episode
        ep_reward = 0.0  # accumulated reward this episode

        while (not done) and steps < int(args.max_steps_per_episode):  # episode interaction loop
            eps = eps_by_step(global_step, float(args.eps_start), float(args.eps_end), int(args.eps_decay_steps))  # compute epsilon at this step

            if random.random() < float(eps):  # explore with probability epsilon
                a = int(random.randint(0, 3))  # random action among 4 move slots
            else:
                st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)  # obs to batched tensor
                a = int(torch.argmax(q(st), dim=1).item())  # greedy action from online Q network

            s2, r, done, info = env.step(a)  # transition environment with chosen action
            buf.add(s, a, float(r), s2, float(done))  # store transition in replay buffer

            s = s2  # advance state
            ep_reward = float(op.add(float(ep_reward), float(r)))  # accumulate reward
            steps += 1  # increment episode step
            global_step += 1  # increment total step counter

            if len(buf) >= int(args.warmup_steps):  # only start learning after buffer has enough data
                for _ in range(int(args.updates_per_step)):  # perform multiple gradient updates per env step
                    sb, ab, rb, s2b, db = buf.sample(batch_size=int(args.batch_size))  # sample replay minibatch

                    sbt = torch.tensor(sb, dtype=torch.float32, device=device)  # batch states tensor
                    abt = torch.tensor(ab, dtype=torch.int64, device=device).unsqueeze(1)  # batch actions tensor with column shape
                    rbt = torch.tensor(rb, dtype=torch.float32, device=device).unsqueeze(1)  # batch rewards tensor
                    s2t = torch.tensor(s2b, dtype=torch.float32, device=device)  # batch next states tensor
                    dbt = torch.tensor(db, dtype=torch.float32, device=device).unsqueeze(1)  # batch done flags tensor

                    qsa = q(sbt).gather(1, abt)  # Q(s,a) for actions actually taken

                    with torch.no_grad():  # no gradients through target computation
                        maxq2 = qt(s2t).max(dim=1, keepdim=True)[0]  # max_a' Q_target(s2, a')
                        target = rbt + float(args.gamma) * maxq2 * (1.0 - dbt)  # Bellman target with terminal masking

                    loss = F.smooth_l1_loss(qsa, target)  # Huber loss for stability

                    opt.zero_grad()  # clear old gradients
                    loss.backward()  # backprop through online Q network
                    torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)  # gradient clipping to avoid exploding updates
                    opt.step()  # apply optimizer update

            if int(global_step) % int(args.target_update_every) == 0:  # periodically update target network
                qt.load_state_dict(q.state_dict())  # sync target weights from online weights
                qt.eval()  # keep target in eval mode

        do_eval = (int(ep) + 1) % int(args.eval_every) == 0  # evaluation trigger based on episode count
        do_save = (int(ep) + 1) % int(args.save_every) == 0  # checkpoint trigger based on episode count

        if do_eval:
            wr = eval_suite(
                q=q,  # online Q network to evaluate
                device=device,  # device for tensor operations
                name_pool=name_pool,  # name pool, not used directly inside eval currently
                suite_pairs=suite_pairs,  # fixed suite of matchup pairs
                n_eval_per_pair=int(args.suite_eval_per_pair),  # episodes per matchup
                seed=int(args.seed) + 12345 + int(ep),  # evaluation seed varies per eval call
                max_steps=int(args.max_steps_per_episode),  # evaluation episode step limit
            )

            rec = {
                "episode": int(ep) + 1,  # one based episode number
                "global_step": int(global_step),  # total steps so far
                "suite_win_rate": float(wr),  # win rate on evaluation suite
                "last_episode_reward": float(ep_reward),  # reward from the most recent training episode
                "seconds": float(time.time() - start_time),  # elapsed wall time since training started
                "device": str(device),  # device string for reproducibility
            }

            with open(log_path, "a", encoding="utf8") as f:
                f.write(json.dumps(rec) + "\n")  # append record as JSON line

            print(rec)  # also print record to console

        if do_save:
            ckpt = out_dir / f"ckpt_ep_{int(ep) + 1}.pt"  # checkpoint filename includes episode count
            meta = {"episode": int(ep) + 1, "global_step": int(global_step), "obs_dim": int(obs_dim)}  # metadata for resume
            save_checkpoint(str(ckpt), q, opt, meta)  # save checkpoint
            print("saved", str(ckpt))  # console confirmation


if __name__ == "__main__":
    main()  # run training when executed as a script
