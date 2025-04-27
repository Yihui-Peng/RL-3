#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
"""

from __future__ import annotations
import numpy as np
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth
import time
from typing import List, Type, Tuple



n_timesteps = 10001
eval_interval = 250
n_repetitions = 20
gamma = 1.0
learning_rate = 0.2
epsilon = 0.1
smooth_window = 9

planning_list = [0, 1, 3, 5]
wind_cases = {
    "stochastic": 0.9,     # wind blows 90 %
    "deterministic": 1.0   # wind always blows → deterministic dynamics
}

# mapping algorithm name to class
agent_map = {
    "dyna": DynaAgent,
    "ps"  : PrioritizedSweepingAgent,
}

# Core training routine

def run_repetition(agent_cls: Type, wind: float, n_planning: int) -> List[float]:
    env = WindyGridworld(wind_proportion=wind)
    eval_env = WindyGridworld(wind_proportion=wind)
    agent = agent_cls(env.n_states, env.n_actions, learning_rate, gamma)

    n_evals = (n_timesteps - 1) // eval_interval + 1
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
    window = min(smooth_window, (len(y)//2)*2+1)
    return smooth(y, window)


def make_plot(title: str, x: np.ndarray, curves: dict, filename: str):
    plotter = LearningCurvePlot(title=title)
    for label, y in curves.items():
        plotter.add_curve(x, smoothed(y), label)
    plotter.save(filename)
    print(f"[saved] {filename}")



# Main experiment 

def experiment():
    steps_axis = np.arange(0, n_timesteps, eval_interval)
    runtime_table = []

    best_curves = {}  # store best performing curve for comparison

    for alg_name, AgentCls in agent_map.items():
        for wind_case, wind_prop in wind_cases.items():

            curves = {}
            best_label, best_curve = None, None
            best_final = -np.inf

            for n_plan in planning_list:
                label = f"K={n_plan}" if n_plan > 0 else "Q-learning"
                print(f"Running {alg_name} | {wind_case} | {label} …")
                mean_curve, avg_rt = run_batch(AgentCls, wind_prop, n_plan)
                curves[label] = mean_curve
                runtime_table.append((alg_name, wind_case, n_plan, avg_rt))

                if mean_curve[-1] > best_final and n_plan > 0:
                    best_final = mean_curve[-1]
                    best_label = label
                    best_curve = mean_curve

            fname = f"{alg_name}_{wind_case}.png"
            make_plot(f"{alg_name.upper()} — {wind_case}", steps_axis, curves, fname)

            if best_curve is not None:
                best_curves[(alg_name, wind_case)] = (best_label, best_curve)

    # comparison plots
    for wind_case in wind_cases.keys():
        comp_curves = {}
        for alg_name in ("dyna", "ps"):
            if (alg_name, wind_case) in best_curves:
                label, curve = best_curves[(alg_name, wind_case)]
                comp_curves[f"{alg_name.upper()} ({label})"] = curve
        comp_curves["Q-learning"] = run_batch(DynaAgent, wind_cases[wind_case], 0)[0]
        fname = f"compare_{wind_case}.png"
        make_plot(f"Best comparison — {wind_case}", steps_axis, comp_curves, fname)

    # runtime table
    print("\nAverage runtime per repetition (s):")
    print("alg\twind\tK\truntime")
    for alg, w, k, rt in runtime_table:
        print(f"{alg}\t{w}\t{k}\t{rt:.3f}")



if __name__ == "__main__":
    experiment()
