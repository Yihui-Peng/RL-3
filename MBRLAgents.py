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

# ADD VISITED STATES ONLY

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
        self.Rsum = np.zeros((n_states, n_actions, n_states)) # s,a,s' - total reward

    def select_action(self, s, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            max_q = np.max(self.Q_sa[s, :])
            max_actions = np.where(self.Q_sa[s, :] == max_q)[0] # select all actions that have the best performance
            return np.random.choice(max_actions) 
        

    def update(self,s,a,r,done,s_next,n_planning_updates):
        # NOTE: done means s!=goal (otherwise no a), thus we update the same way (Q_sa(a) will be 0)
        # Direct Learning
        
        self.Q_sa[s, a] += self.alpha * ((r + self.gamma * np.max(self.Q_sa[s_next])) - self.Q_sa[s, a]) # greedy action selection, as evaluation is done that way

        # Indirect Learning
            # Model Learning - only Rsum and n needed to understand all of model - Transition and reward function
        self.n[s, a, s_next] += 1
        self.Rsum[s, a, s_next] += r

            # Planning         
        for plan in range(n_planning_updates):
            # s - plan
            visited_states = [s for s in range(self.n.shape[0]) if np.sum(self.n[s].flatten()) > 0]
            p_s = np.random.choice(visited_states) # There exists at least one state, so no if condition
                
            # a - plan
            visited_actions = [act for act in range(len(self.n[p_s])) if np.sum(self.n[p_s, act]) != 0] # a is the rows of n[s], so range of length are the actions themselves
            p_a = np.random.choice(visited_actions) # All states have the same number of actions

            # s' and r - plan
            p_s_next = self.transition_to(p_s, p_a)
            p_r = self.reward_estimate(p_s, p_a, p_s_next)
            self.Q_sa[p_s, p_a] += self.alpha * ((p_r + self.gamma * np.max(self.Q_sa[p_s_next])) - self.Q_sa[p_s, p_a]) # Indirect update

    def transition_to(self, s, a):
        return np.random.choice(range(self.n_states), p=self.n[s, a]/np.sum(self.n[s, a])) # Select next state given the transition probabilities
                                                                                        # transition prob are found by dividing number of visits by all visits 
    
    # This can be optimized by only considering the 4 adjacent possitions

    # def transition_estimate(self, s, a, s_next):
    #     total = np.sum(self.n[s, a])
    #     return self.n[s, a, s_next] / total
        
    def reward_estimate(self, s, a, s_next):
        total = self.n[s, a, s_next]
        return self.Rsum[s, a, s_next] / total

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return        


class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
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
        # NOTE: done means s!=goal (otherwise no a), thus we update the same way (Q_sa(a) will be 0)
        # Direct Learning
        self.update_queue(s,a,r,s_next)

        # Indirect Learning
            # Model Learning - only Rsum and n needed to understand all of model - Transition and reward function
        
        # print(f"UPDATE: s = {s}  a = {a}  s_next = {s_next}  r = {r}")
        self.n[s, a, s_next] += 1
        
        # self.Rsum[s, a, s_next] += r
        self.Rsum[s, a, s_next] += (r - self.Rsum[s, a, s_next]) / self.n[s, a, s_next]

            # Planning         
        counter = 0
        # print("\n NEW START \n")
        # print(sum(self.n.flatten()))
        while not self.queue.empty() and counter < n_planning_updates: # if no more q or already reached the maximum number of updates through planning
            # print(f"ONCE: {self.queue.queue}")
            # print(sum(self.n.flatten()))

                # Update
            _, (p_s, p_a) = self.queue.get()
            p_s_next = self.transition_to(p_s, p_a)
            p_r = self.reward_estimate(p_s, p_a, p_s_next)
            self.Q_sa[p_s, p_a] += self.alpha * ((p_r + self.gamma * np.max(self.Q_sa[p_s_next])) - self.Q_sa[p_s, p_a]) # Indirect update
                
                # Backtrack - b is the backtracking state
            for b_p_s in range(self.n_states): # backtracking_planning_state
                for b_p_a in range(self.n_actions):
                    if self.n[b_p_s, b_p_a, p_s] > 0: # as states are represented atomically, we can use s to access the number of times other s and a lead to the current s
                        # print(f"BACKTRACKING: b_p_s = {b_p_s}  b_p_a = {b_p_a} p_s = {p_s}")
                        # Update backtrack
                        b_p_r = self.reward_estimate(b_p_s, b_p_a, p_s)
                        self.update_queue(b_p_s, b_p_a, b_p_r, p_s)
            counter +=1

    def update_queue(self, s, a, r, s_next):
        difference = abs(r + self.gamma * np.max(self.Q_sa[s_next]) - self.Q_sa[s, a]) # difference between the current and the new value 
        if difference > self.priority_cutoff: # abs of difference because change can be negative
            self.queue.put((-difference, (s, a))) # Priority queue prioritizes the smallest values, so larger differences are smaller values when turned negative, thus has a higher priority


    def transition_to(self, s, a):
        return np.random.choice(self.n_states, p=self.n[s, a]/np.sum(self.n[s, a])) # Select next state given the transition probabilities
                                                                                           # transition prob are found by dividing number of visits by all visits 
        
    def reward_estimate(self, s, a, s_next):
        # total = self.n[s, a, s_next]
        # return self.Rsum[s, a, s_next] / total
        return self.Rsum[s, a, s_next]
    
    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return        

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