#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld

class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma): # epsilon at 1 = random policy
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        # TO DO: Initialize relevant elements

        # n over here, represents the number of times action a has been taken in state s
        self.Q_sa = np.zeros((n_states, n_actions))
        self.n = np.zeros((n_states, n_actions, n_states), dtype=int) # s,a,s' - number of visits
        self.Rsum = np.zeros((n_states, n_actions, n_states)) # s,a,s' - total reward
        self.model_sa = set()  # Track observed (s,a) pairs

    def select_action(self, s, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            max_q = np.max(self.Q_sa[s, :])
            max_actions = np.where(self.Q_sa[s, :] == max_q)[0] # select all actions that have the best performance
            return np.random.choice(max_actions) 
        

    def update(self,s,a,r,done,s_next,n_planning_updates):
        self.n[s, a, s_next] += 1
        self.Rsum[s, a, s_next] += r
        self.model_sa.add((s, a))
        
        # Direct Q-learning update
        max_next_Q = np.max(self.Q_sa[s_next]) if not done else 0.0
        td_error = r + self.gamma * max_next_Q - self.Q_sa[s, a]
        self.Q_sa[s, a] += self.learning_rate * td_error
        
        # Planning updates
        for _ in range(n_planning_updates):
            if not self.model_sa:
                continue
            # Randomly select a (s_plan, a_plan) from observed
            sa_list = list(self.model_sa)
            s_plan, a_plan = sa_list[np.random.randint(len(sa_list))]
            
            # Sample s_plan_next from model
            total_transitions = np.sum(self.n[s_plan, a_plan])
            if total_transitions == 0:
                continue
            s_possible = np.where(self.n[s_plan, a_plan] > 0)[0]
            probs = self.n[s_plan, a_plan, s_possible] / total_transitions
            s_plan_next = np.random.choice(s_possible, p=probs)
            r_plan = self.Rsum[s_plan, a_plan, s_plan_next] / self.n[s_plan, a_plan, s_plan_next]
            
            # Update Q-value
            max_next_plan_Q = np.max(self.Q_sa[s_plan_next])
            td_error_plan = r_plan + self.gamma * max_next_plan_Q - self.Q_sa[s_plan, a_plan]
            self.Q_sa[s_plan, a_plan] += self.learning_rate * td_error_plan


    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []
        for _ in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            done = False
            for _ in range(max_episode_length):
                a = np.argmax(self.Q_sa[s])
                s_next, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                s = s_next
            returns.append(R_ep)
        return np.mean(returns)    


class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue()
        # TO DO: Initialize relevant elements
        
        self.Q_sa = np.zeros((n_states, n_actions))
        self.n = np.zeros((n_states, n_actions, n_states))
        self.Rsum = np.zeros((n_states, n_actions, n_states))

        
    def select_action(self, s, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            max_q = np.max(self.Q_sa[s, :])
            max_actions = np.where(self.Q_sa[s, :] == max_q)[0]
            return np.random.choice(max_actions) 
        
    def update(self,s,a,r,done,s_next,n_planning_updates):
        self.n[s, a, s_next] += 1
        self.Rsum[s, a, s_next] += r
        
        # Calculate priority for current (s,a)
        max_next_Q = np.max(self.Q_sa[s_next]) if not done else 0.0
        td_error = r + self.gamma * max_next_Q - self.Q_sa[s, a]
        p = abs(td_error)
        if p > self.priority_cutoff:
            self.queue.put((-p, (s, a)))
        
        # Perform planning updates
        for _ in range(n_planning_updates):
            if self.queue.empty():
                break
            _, (s_plan, a_plan) = self.queue.get()
            
            # Sample from model
            total_transitions = np.sum(self.n[s_plan, a_plan])
            if total_transitions == 0:
                continue
            s_possible = np.where(self.n[s_plan, a_plan] > 0)[0]
            probs = self.n[s_plan, a_plan, s_possible] / total_transitions
            s_plan_next = np.random.choice(s_possible, p=probs)
            r_plan = self.Rsum[s_plan, a_plan, s_plan_next] / self.n[s_plan, a_plan, s_plan_next]
            
            # Update Q-value
            max_next_plan_Q = np.max(self.Q_sa[s_plan_next])
            td_error_plan = r_plan + self.gamma * max_next_plan_Q - self.Q_sa[s_plan, a_plan]
            self.Q_sa[s_plan, a_plan] += self.learning_rate * td_error_plan
            
            # Find all (s_prev, a_prev) leading to s_plan_next
            for s_prev in range(self.n_states):
                for a_prev in range(self.n_actions):
                    if self.n[s_prev, a_prev, s_plan_next] > 0:
                        r_prev = self.Rsum[s_prev, a_prev, s_plan_next] / self.n[s_prev, a_prev, s_plan_next]
                        p_prev = abs(r_prev + self.gamma * np.max(self.Q_sa[s_plan_next]) - self.Q_sa[s_prev, a_prev])
                        if p_prev > self.priority_cutoff:
                            self.queue.put((-p_prev, (s_prev, a_prev)))

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []
        for _ in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            done = False
            for _ in range(max_episode_length):
                a = np.argmax(self.Q_sa[s])
                s_next, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                s = s_next
            returns.append(R_ep)
        return np.mean(returns)     

def test():

    n_timesteps = 10001
    gamma = 1.0

    # Algorithm parameters
    policy = 'ps' # dyna' or 'ps' 
    epsilon = 0.1
    learning_rate = 0.2
    n_planning_updates = 3

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001
    
    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
    elif policy == 'ps':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))
    
    # Prepare for running
    s = env.reset()  
    continuous_mode = False
    
    for t in range(n_timesteps):            
        # Select action, transition, update policy
        a = pi.select_action(s,epsilon)
        s_next,r,done = env.step(a)
        pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates)
        
        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)
            
        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next
            
    
if __name__ == '__main__':
    test()
