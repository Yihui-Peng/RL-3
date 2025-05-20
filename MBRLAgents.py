import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld


class DynaAgent:

    def __init__(self, n_states: int, n_actions: int, learning_rate: float, gamma: float):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = gamma
        # TO DO: Initialize relevant elements

        # n over here, represents the number of times action a has been taken in state s
        self.Q_sa = np.zeros((n_states, n_actions))
        self.n = np.zeros((n_states, n_actions, n_states)) # s,a,s' - number of visits
        self.Rsum = np.zeros((n_states, n_actions, n_states)) # s,a,s' - total reward


    def select_action(self, s: int, epsilon: float):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            max_q = np.max(self.Q_sa[s])
            max_actions = np.where(self.Q_sa[s] == max_q)[0]
            return np.random.choice(max_actions)


    def update(self, s: int, a: int, r: float, done: bool, s_next: int, n_planning_updates: int):
        target = r if done else r + self.gamma * np.max(self.Q_sa[s_next])
        td_error = target - self.Q_sa[s, a]
        self.Q_sa[s, a] += self.alpha * td_error

        # Model learning (update counts and reward sums) -------------------
        self.n[s, a, s_next] += 1
        self.Rsum[s, a, s_next] += r

        # Planning updates -------------------------------------------------
        # Pre‑compute sets of visited (s,a) for uniform sampling
        visited_state_mask = np.sum(self.n, axis=(1, 2)) > 0
        visited_states = np.where(visited_state_mask)[0]
        if visited_states.size == 0:
            return  # nothing to plan yet

        # I put the transition_to and reward_estimate inside the planning loop ----------------
        for _ in range(n_planning_updates):
            # 1) sample previously observed state
            p_s = np.random.choice(visited_states)
            # 2) sample action previously taken in that state
            visited_actions = np.where(np.sum(self.n[p_s], axis=1) > 0)[0]
            p_a = np.random.choice(visited_actions)
            # 3) sample next state from learned transition model
            trans_counts = self.n[p_s, p_a]
            total = trans_counts.sum()
            if total == 0:
                continue  # should not happen but be safe
            trans_prob = trans_counts / total
            p_s_next = np.random.choice(len(trans_prob), p=trans_prob)
            # 4) estimate reward (mean)
            p_r = self.Rsum[p_s, p_a, p_s_next] / self.n[p_s, p_a, p_s_next]
            # 5) Q update
            target_model = p_r + self.gamma * np.max(self.Q_sa[p_s_next])
            self.Q_sa[p_s, p_a] += self.alpha * (target_model - self.Q_sa[p_s, p_a])


    def evaluate(self, eval_env: WindyGridworld, n_eval_episodes: int = 30, max_episode_length: int = 100):
        returns = []
        for _ in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for _ in range(max_episode_length):
                a = np.argmax(self.Q_sa[s])  # greedy action
                s, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
            returns.append(R_ep)
        return np.mean(returns)



class PrioritizedSweepingAgent:

    def __init__(self, n_states: int, n_actions: int, learning_rate: float, gamma: float, priority_cutoff: float = 0.01):
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


    def select_action(self, s: int, epsilon: float):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            max_q = np.max(self.Q_sa[s])
            max_actions = np.where(self.Q_sa[s] == max_q)[0]
            return np.random.choice(max_actions)


    def update(self, s: int, a: int, r: float, done: bool, s_next: int, n_planning_updates: int):
        # Compute priority p (Direct temporal-difference update)-------------------------------------------
        target = r if done else r + self.gamma * np.max(self.Q_sa[s_next])
        td_error = target - self.Q_sa[s, a]
        self.Q_sa[s, a] += self.alpha * td_error

        # Model learning (update counts and reward sums) -------------------
        self.n[s, a, s_next] += 1
        self.Rsum[s, a, s_next] += r

        # Insert (s,a) into priority queue if priority large ----------------
        priority = abs(td_error)
        if priority > self.priority_cutoff:
            # Negative priority because PriorityQueue pops smallest first
            self.queue.put((-priority, (s, a)))

        #  Planning loop -----------------------------------------
        planning_counter = 0
        while not self.queue.empty() and planning_counter < n_planning_updates:
            _, (p_s, p_a) = self.queue.get()

            # Sample model transition & reward
            trans_counts = self.n[p_s, p_a]
            total = trans_counts.sum()
            if total == 0:
                planning_counter += 1
                continue  # no data
            
            p_s_next = np.random.choice(len(trans_counts), p=trans_counts / total)
            p_r = self.Rsum[p_s, p_a, p_s_next] / self.n[p_s, p_a, p_s_next]

            # TD update for sampled pair
            target_model = p_r + self.gamma * np.max(self.Q_sa[p_s_next])
            td_model = target_model - self.Q_sa[p_s, p_a]
            self.Q_sa[p_s, p_a] += self.alpha * td_model

            # Back‑propagate to predecessors of p_s
            # Environment is small (70 states), scan all; optimisation possible for larger tasks.
            for b_s in range(self.n_states):
                for b_a in range(self.n_actions):
                    if self.n[b_s, b_a, p_s] == 0:
                        continue
                    b_r = self.Rsum[b_s, b_a, p_s] / self.n[b_s, b_a, p_s]
                    priority_b = abs(b_r + self.gamma * np.max(self.Q_sa[p_s]) - self.Q_sa[b_s, b_a])
                    if priority_b > self.priority_cutoff:
                        self.queue.put((-priority_b, (b_s, b_a)))

            planning_counter += 1


    def evaluate(self, eval_env: WindyGridworld, n_eval_episodes: int = 30, max_episode_length: int = 100) -> float:
        returns = []
        for _ in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for _ in range(max_episode_length):
                a = np.argmax(self.Q_sa[s])
                s, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
            returns.append(R_ep)
        return np.mean(returns)



# Bonus part
class DynaUCBAgent(DynaAgent):
    """
    Shares exactly the same model-learning and planning machinery as `DynaAgent`,
    but replaces ε-greedy exploration with an Upper-Confidence-Bound (UCB) rule:

        a* = argmax_a  Q(s,a) + c · sqrt( ln N(s) / N(s,a) )

    where
        N(s)   = ∑_a N(s,a)               # total visits to state s
        N(s,a) = ∑_{s'} n(s,a,s')         # visits to state–action pair (s,a)
    The counts come straight from the empirical transition tally `self.n`.
    """
    
    def __init__(self, n_states: int, n_actions: int, learning_rate: float, gamma: float, c: float = 2.0):
        super().__init__(n_states, n_actions, learning_rate, gamma)
        self.c = c

    # the ε parameter is ignored here –– we use pure UCB.
    def select_action(self, s: int, _unused: float = None) -> int:
        """Pick an action via UCB (untried actions are forced to be tried first)."""
        # Empirical visit counts:
        # N(s,a) = Σ_{s'} n(s,a,s'),  shape = (n_actions,)
        N_sa = self.n[s].sum(axis=1)

        # If some action in this state has never been tried, explore it immediately.
        if (N_sa == 0).any():
            return np.random.choice(np.flatnonzero(N_sa == 0))

        N_s = N_sa.sum()
        ucb = self.Q_sa[s] + self.c * np.sqrt(np.log(N_s) / N_sa)
        max_ucb = np.max(ucb)
        return np.random.choice(np.flatnonzero(ucb == max_ucb))



def test():

    n_timesteps = 10001
    gamma = 1.0

    # Algorithm parameters
    policy = 'dyna' # or 'ps' 
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
