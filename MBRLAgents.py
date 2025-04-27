#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld


class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma): # epsilon at 1 = random policy
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = gamma
        # TO DO: Initialize relevant elements

        # n over here, represents the number of times action a has been taken in state s
        self.Q_sa = np.zeros((n_states, n_actions))
        self.n = np.zeros((n_states, n_actions, n_states)) # s,a,s' - number of visits
        self.R_sum = np.zeros((n_states, n_actions, n_states)) # s,a,s' - total reward

        # Memory of observed (s,a) pairs for planning
        self.observed_sa = []              # list[(s,a)]
        self._seen_mask  = np.zeros((n_states, n_actions), dtype=bool)

    def select_action(self, s, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            max_q = np.max(self.Q_sa[s, :])
            max_actions = np.where(self.Q_sa[s, :] == max_q)[0] # select all actions that have the best performance
            return np.random.choice(max_actions) 

    def update(self, s, a, r, done, s_next, n_planning_updates):
        # Model update
        self.n[s, a, s_next] += 1
        self.R_sum[s, a, s_next]   += r

        if not self._seen_mask[s, a]:
            self.observed_sa.append((s, a))
            self._seen_mask[s, a] = True

        # Direct TD update on real transition
        target = r if done else r + self.gamma * self.Q_sa[s_next].max()
        self.Q_sa[s, a] += self.alpha * (target - self.Q_sa[s, a])

        # Planning updates
        if n_planning_updates == 0 or not self.observed_sa:
            return

        for _ in range(n_planning_updates):
            s_p, a_p = self.observed_sa[np.random.randint(len(self.observed_sa))]

            counts = self.n[s_p, a_p]
            total  = counts.sum()
            if total == 0:
                continue  # should not happen, but be safe.
            probs = counts / total
            s_p_next = np.random.choice(self.n_states, p=probs)

            r_hat = self.R_sum[s_p, a_p, s_p_next] / max(1, counts[s_p_next])
            target_p = r_hat + self.gamma * self.Q_sa[s_p_next].max()
            self.Q_sa[s_p, a_p] += self.alpha * (target_p - self.Q_sa[s_p, a_p])


    def evaluate(self, eval_env: WindyGridworld, n_eval_episodes=30, max_episode_length=100):
        returns = []
        for _ in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for _ in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, reward, done = eval_env.step(a)
                R_ep += reward
                if done:
                    break
                s = s_prime
            returns.append(R_ep)
        return float(np.mean(returns))

class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, priority_cutoff: float = 1e-2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue()
        # TO DO: Initialize relevant elements

        self.Q_sa = np.zeros((n_states, n_actions))
        self.n = np.zeros((n_states, n_actions, n_states))
        self.R_sum = np.zeros((n_states, n_actions, n_states))

        # Predecessor graph: predecessors[s′] = set((s,a),…)
        self.predecessors = [set() for _ in range(n_states)]


    def select_action(self, s: int, epsilon: float) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            max_q = np.max(self.Q_sa[s, :])
            max_actions = np.where(self.Q_sa[s, :] == max_q)[0]
            return np.random.choice(max_actions) 


    def update(self, s, a, r, done, s_next, n_planning_updates):
        # Model update
        self.n[s, a, s_next] += 1
        self.R_sum[s, a, s_next]   += r
        self.predecessors[s_next].add((s, a))

        # Push current (s,a) onto queue with priority 
        target = r if done else r + self.gamma * self.Q_sa[s_next].max()
        p = abs(target - self.Q_sa[s, a])
        if p > self.priority_cutoff:
            self.queue.put((-p, (s, a)))  # negative because PriorityQueue is min-heap

        # Planning
        for _ in range(n_planning_updates):
            if self.queue.empty():
                break
            _, (s_p, a_p) = self.queue.get()

            # Sample s′ from model of (s_p,a_p)
            counts = self.n[s_p, a_p]
            total = counts.sum()
            if total == 0:
                continue
            probs = counts / total
            s_p_next = np.random.choice(self.n_states, p=probs)
            r_hat = self.R_sum[s_p, a_p, s_p_next] / max(1, counts[s_p_next])

            # Q update
            target_p = r_hat + self.gamma * self.Q_sa[s_p_next].max()
            delta = target_p - self.Q_sa[s_p, a_p]
            self.Q_sa[s_p, a_p] += self.alpha * delta

            # Propagate to predecessors of s_p
            for s_pre, a_pre in self.predecessors[s_p]:
                counts_pa = self.n[s_pre, a_pre, s_p]
                if counts_pa == 0:
                    continue
                r_pre = self.R_sum[s_pre, a_pre, s_p] / counts_pa
                pred_target = r_pre + self.gamma * self.Q_sa[s_p].max()
                p_pre = abs(pred_target - self.Q_sa[s_pre, a_pre])
                if p_pre > self.priority_cutoff:
                    self.queue.put((-p_pre, (s_pre, a_pre)))


    def evaluate(self, eval_env: WindyGridworld, n_eval_episodes=30,
                 max_episode_length=100) -> float:
        returns = []
        for _ in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for _ in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, reward, done = eval_env.step(a)
                R_ep += reward
                if done:
                    break
                s = s_prime
            returns.append(R_ep)
        return float(np.mean(returns))


def test():
    n_timesteps = 10001
    gamma = 1.0

    policy = 'ps'      # 'dyna' or 'ps'
    epsilon = 0.1
    learning_rate = 0.2
    n_planning_updates = 3

    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001

    env = WindyGridworld()
    if policy == 'dyna':
        agent = DynaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    elif policy == 'ps':
        agent = PrioritizedSweepingAgent(env.n_states, env.n_actions,
                                         learning_rate, gamma)
    else:
        raise KeyError('Unknown policy')

    s = env.reset()
    continuous = False
    for _ in range(n_timesteps):
        a = agent.select_action(s, epsilon)
        s_next, r, done = env.step(a)
        agent.update(s, a, r, done, s_next, n_planning_updates)

        if plot:
            env.render(Q_sa=agent.Q_sa, plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)

        if not continuous:
            key = input("<Enter> next, 'c' continuous: ")
            continuous = (key.lower() == 'c')
        s = env.reset() if done else s_next

if __name__ == '__main__':
    test()