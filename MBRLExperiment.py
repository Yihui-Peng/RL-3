from __future__ import annotations
import time
from typing import List, Tuple, Type

import numpy as np
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth


n_timesteps = 10001
eval_interval = 250
n_repetitions = 20

gamma = 1.0
learning_rate = 0.2
epsilon = 0.1

smooth_window = 11

n_planning_updates = [1, 3, 5]
wind_cases = {
    "stochastic": 0.9,
    "deterministic": 1.0
}

agent_map = {
    "dyna": DynaAgent,
    "ps": PrioritizedSweepingAgent,
}


def run_repetition(agent_cls: Type, wind: float, n_planning: int) -> List[float]:
    env = WindyGridworld(wind_proportion=wind)
    eval_env = WindyGridworld(wind_proportion=wind)
    agent = agent_cls(env.n_states, env.n_actions, learning_rate, gamma)

    n_evals = (n_timesteps - 1) // eval_interval + 1  # include t=0
    curve = np.empty(n_evals, dtype=float)
    eval_idx = 0

    curve[eval_idx] = agent.evaluate(eval_env)
    eval_idx += 1

    s = env.reset()
    for t in range(1, n_timesteps):
        a = agent.select_action(s, epsilon)
        s_next, r, done = env.step(a)
        agent.update(s, a, r, done, s_next, n_planning)
        s = env.reset() if done else s_next

        if t % eval_interval == 0:
            curve[eval_idx] = agent.evaluate(eval_env)
            eval_idx += 1

    return curve.tolist()


def run_batch(agent_cls: Type, wind: float, n_planning: int) -> Tuple[np.ndarray, float]:
    all_curves = []
    tic = time.time()
    for _ in range(n_repetitions):
        all_curves.append(run_repetition(agent_cls, wind, n_planning))
    runtime = (time.time() - tic) / n_repetitions
    mean_curve = np.mean(all_curves, axis=0)
    return mean_curve, runtime


def smoothed(y: np.ndarray) -> np.ndarray:
    window = min(smooth_window, (len(y) // 2) * 2 + 1)  # keep odd & <= len
    return smooth(y, window)


def make_plot(title: str, x: np.ndarray, curves: dict[str, np.ndarray], filename: str):
    plotter = LearningCurvePlot(title=title)
    for label, y in curves.items():
        plotter.add_curve(x, smoothed(y), label)
    plotter.save(filename)
    print(f"[saved] {filename}")


def experiment():
    steps_axis = np.arange(0, n_timesteps, eval_interval)

    runtime_table: list[tuple[str, str, int, float]] = []  # alg, wind, K, sec
    best_curves: dict[tuple[str, str], tuple[str, np.ndarray]] = {}
    baseline_curves: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Pure Q‑learning baseline → DynaAgent with K=0 planning
    for wind_case, wind_prop in wind_cases.items():
        print(f"Running Q‑learning baseline | {wind_case}")
        q_curve, q_rt = run_batch(DynaAgent, wind_prop, 0)
        baseline_curves[wind_case] = q_curve
        runtime_table.append(("q", wind_case, 0, q_rt))

    # ------------------------------------------------------------------
    # Structured sweeps for Dyna & Prioritized Sweeping
    for alg_key, AgentCls in agent_map.items():
        for wind_case, wind_prop in wind_cases.items():
            curves: dict[str, np.ndarray] = {
                "n_planning_updates=0": baseline_curves[wind_case]
            }
            best_label, best_curve = None, None
            best_final = -np.inf

            for n_plan in n_planning_updates:
                label = f"n_planning_updates={n_plan}"
                print(f"Running {alg_key} | {wind_case} | {label} …")
                mean_curve, avg_rt = run_batch(AgentCls, wind_prop, n_plan)
                curves[label] = mean_curve
                runtime_table.append((alg_key, wind_case, n_plan, avg_rt))

                # keep track of best performer (final evaluation)
                if mean_curve[-1] > best_final:
                    best_final = mean_curve[-1]
                    best_label = label
                    best_curve = mean_curve

            # save per‑algorithm plot
            fname = f"{alg_key}_{wind_case}.png"
            make_plot(f"Curves with {alg_key.upper()} — {wind_case}, wind={wind_prop}", steps_axis, curves, fname)

            if best_curve is not None:
                best_curves[(alg_key, wind_case)] = (best_label, best_curve)

    # ------------------------------------------------------------------
    # Final comparison plots (best Dyna vs best PS vs baseline)
    for wind_case in wind_cases.keys():
        comp_curves: dict[str, np.ndarray] = {
            "Q‑learning": baseline_curves[wind_case]
        }
        for alg_key in ("dyna", "ps"):
            if (alg_key, wind_case) in best_curves:
                label, curve = best_curves[(alg_key, wind_case)]
                comp_curves[f"{alg_key.upper()} ({label})"] = curve
        fname = f"compare_{wind_case}.png"
        make_plot(f"Best comparison — {wind_case}, wind={wind_cases[wind_case]}", steps_axis, comp_curves, fname)

    # ------------------------------------------------------------------
    # Print runtime summary table
    print("\nAverage runtime per repetition (s):")
    print("alg\twind\tn_planning_updates\truntime")
    for alg, wcase, k, rt in runtime_table:
        print(f"{alg}\t{wcase}\t{k}\t{rt:.3f}")


if __name__ == "__main__":
    experiment()


# To run the code:
# python -m venv .venv
# .\.venv\Scripts\Activate
# python -m pip install --upgrade pip
# pip install numpy matplotlib
# pip install scipy