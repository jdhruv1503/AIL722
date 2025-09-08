import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'or_gym'))
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv
import matplotlib.pyplot as plt
from scipy.stats import mode

class ValueIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-4):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        
        # State space: (current_weight, item_idx, item_weight, item_value)
        # Action space: 0 (reject), 1 (accept)
        self.max_weight = env.max_weight
        self.num_items = env.N
        self.max_item_weight = 100  # From env initialization
        self.max_item_value = 100   # From env initialization
        
        # Initialize value function
        self.V = np.zeros((self.max_weight + 1, self.num_items, self.max_item_weight + 1, self.max_item_value + 1))
        
    def get_reward_and_done(self, current_weight, item_idx, action):
        """Get reward and done status for a state-action pair"""
        if action == 0:  # Reject
            return 0, False
        else:  # Accept
            item_weight = self.env.item_weights[item_idx]
            if current_weight + item_weight <= self.max_weight:
                return self.env.item_values[item_idx], False
            else:
                return 0, True  # Overweight, episode ends
    
    def value_iteration(self, max_iterations=1000):
        """Run value iteration algorithm"""
        for iteration in range(max_iterations):
            delta = 0
            V_old = self.V.copy()
            
            for w in range(self.max_weight + 1):
                for item_idx in range(self.num_items):
                    item_weight = self.env.item_weights[item_idx]
                    item_value = self.env.item_values[item_idx]
                    
                    # Skip if item weight exceeds remaining capacity
                    if item_weight > self.max_weight - w:
                        continue
                    
                    # Calculate values for both actions
                    # Action 0: Reject
                    reject_reward, reject_done = self.get_reward_and_done(w, item_idx, 0)
                    if not reject_done:
                        reject_value = reject_reward + self.gamma * V_old[w, (item_idx + 1) % self.num_items, item_weight, item_value]
                    else:
                        reject_value = reject_reward
                    
                    # Action 1: Accept
                    accept_reward, accept_done = self.get_reward_and_done(w, item_idx, 1)
                    if not accept_done:
                        new_weight = w + item_weight
                        accept_value = accept_reward + self.gamma * V_old[new_weight, (item_idx + 1) % self.num_items, item_weight, item_value]
                    else:
                        accept_value = accept_reward
                    
                    # Update value function
                    self.V[w, item_idx, item_weight, item_value] = max(reject_value, accept_value)
                    delta = max(delta, abs(self.V[w, item_idx, item_weight, item_value] - V_old[w, item_idx, item_weight, item_value]))
            
            if delta < self.epsilon:
                break
        
        return iteration + 1
    
    def get_action(self, state):
        """Get optimal action for a given state"""
        if isinstance(state, dict):
            state = state['state']
        
        current_weight, item_idx, item_weight, item_value = state
        
        # Calculate values for both actions
        # Action 0: Reject
        reject_reward, reject_done = self.get_reward_and_done(current_weight, item_idx, 0)
        if not reject_done:
            reject_value = reject_reward + self.gamma * self.V[current_weight, (item_idx + 1) % self.num_items, item_weight, item_value]
        else:
            reject_value = reject_reward
        
        # Action 1: Accept
        accept_reward, accept_done = self.get_reward_and_done(current_weight, item_idx, 1)
        if not accept_done:
            new_weight = current_weight + item_weight
            accept_value = accept_reward + self.gamma * self.V[new_weight, (item_idx + 1) % self.num_items, item_weight, item_value]
        else:
            accept_value = accept_reward
        
        return 1 if accept_value > reject_value else 0

class PolicyIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-4, eval_iterations=1000):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.eval_iterations = eval_iterations
        
        # State space: (current_weight, item_idx, item_weight, item_value)
        self.max_weight = env.max_weight
        self.num_items = env.N
        self.max_item_weight = 100
        self.max_item_value = 100
        
        # Initialize policy and value function
        self.policy = np.zeros((self.max_weight + 1, self.num_items, self.max_item_weight + 1, self.max_item_value + 1), dtype=int)
        self.V = np.zeros((self.max_weight + 1, self.num_items, self.max_item_weight + 1, self.max_item_value + 1))
        
    def get_reward_and_done(self, current_weight, item_idx, action):
        """Get reward and done status for a state-action pair"""
        if action == 0:  # Reject
            return 0, False
        else:  # Accept
            item_weight = self.env.item_weights[item_idx]
            if current_weight + item_weight <= self.max_weight:
                return self.env.item_values[item_idx], False
            else:
                return 0, True
    
    def policy_evaluation(self):
        """Evaluate current policy"""
        for _ in range(self.eval_iterations):
            delta = 0
            V_old = self.V.copy()
            
            for w in range(self.max_weight + 1):
                for item_idx in range(self.num_items):
                    item_weight = self.env.item_weights[item_idx]
                    item_value = self.env.item_values[item_idx]
                    
                    if item_weight > self.max_weight - w:
                        continue
                    
                    action = self.policy[w, item_idx, item_weight, item_value]
                    reward, done = self.get_reward_and_done(w, item_idx, action)
                    
                    if not done:
                        if action == 0:  # Reject
                            next_value = self.V[w, (item_idx + 1) % self.num_items, item_weight, item_value]
                        else:  # Accept
                            new_weight = w + item_weight
                            next_value = self.V[new_weight, (item_idx + 1) % self.num_items, item_weight, item_value]
                        
                        self.V[w, item_idx, item_weight, item_value] = reward + self.gamma * next_value
                    else:
                        self.V[w, item_idx, item_weight, item_value] = reward
                    
                    delta = max(delta, abs(self.V[w, item_idx, item_weight, item_value] - V_old[w, item_idx, item_weight, item_value]))
            
            if delta < self.epsilon:
                break
    
    def policy_improvement(self):
        """Improve policy based on current value function"""
        policy_stable = True
        
        for w in range(self.max_weight + 1):
            for item_idx in range(self.num_items):
                item_weight = self.env.item_weights[item_idx]
                item_value = self.env.item_values[item_idx]
                
                if item_weight > self.max_weight - w:
                    continue
                
                old_action = self.policy[w, item_idx, item_weight, item_value]
                
                # Calculate values for both actions
                # Action 0: Reject
                reject_reward, reject_done = self.get_reward_and_done(w, item_idx, 0)
                if not reject_done:
                    reject_value = reject_reward + self.gamma * self.V[w, (item_idx + 1) % self.num_items, item_weight, item_value]
                else:
                    reject_value = reject_reward
                
                # Action 1: Accept
                accept_reward, accept_done = self.get_reward_and_done(w, item_idx, 1)
                if not accept_done:
                    new_weight = w + item_weight
                    accept_value = accept_reward + self.gamma * self.V[new_weight, (item_idx + 1) % self.num_items, item_weight, item_value]
                else:
                    accept_value = accept_reward
                
                # Update policy
                self.policy[w, item_idx, item_weight, item_value] = 1 if accept_value > reject_value else 0
                
                if self.policy[w, item_idx, item_weight, item_value] != old_action:
                    policy_stable = False
        
        return policy_stable
    
    def run_policy_iteration(self, max_iterations=1000):
        """Run policy iteration algorithm"""
        for iteration in range(max_iterations):
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            
            if policy_stable:
                break
        
        return iteration + 1
    
    def get_action(self, state):
        """Get action from current policy"""
        if isinstance(state, dict):
            state = state['state']
        
        current_weight, item_idx, item_weight, item_value = state
        return self.policy[current_weight, item_idx, item_weight, item_value]

def evaluate_policy(env, policy, num_episodes=5, seeds=None):
    """Evaluate a policy over multiple episodes"""
    if seeds is None:
        seeds = range(num_episodes)
    
    rewards = []
    knapsack_values = []
    
    for seed in seeds:
        env.set_seed(seed)
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < env.step_limit:
            action = policy.get_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        rewards.append(total_reward)
        knapsack_values.append(total_reward)  # In this case, reward = knapsack value
    
    return np.mean(rewards), np.std(rewards), knapsack_values

def plot_knapsack_values(knapsack_values, title="Knapsack Values"):
    """Plot knapsack values over episodes"""
    plt.figure(figsize=(10, 6))
    plt.plot(knapsack_values, marker='o')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Knapsack Value')
    plt.grid(True)
    plt.show()

def plot_value_function_heatmap(V, title="Value Function Heatmap"):
    """Plot heatmap of value function"""
    # Create a simplified 2D representation
    # Use current_weight as y-axis and item_value as x-axis
    weights = np.arange(V.shape[0])
    values = np.arange(V.shape[3])
    
    # Average over item_idx and item_weight dimensions
    V_2d = np.mean(V, axis=(1, 2))
    
    plt.figure(figsize=(12, 8))
    plt.imshow(V_2d, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('Item Value')
    plt.ylabel('Current Weight')
    plt.show()

def main():
    print("=== Online Knapsack Problem Analysis ===")
    
    # Test with different seeds
    seeds = [0, 1, 2, 3, 4]
    
    # Value Iteration
    print("Running Value Iteration...")
    vi_results = []
    for seed in seeds:
        env = OnlineKnapsackEnv()
        env.set_seed(seed)
        
        vi_solver = ValueIterationOnlineKnapsack(env)
        iterations = vi_solver.value_iteration()
        
        mean_reward, std_reward, knapsack_values = evaluate_policy(env, vi_solver, num_episodes=1, seeds=[seed])
        vi_results.append((mean_reward, std_reward, knapsack_values, vi_solver.V))
        
        print(f"Seed {seed}: {iterations} iterations, Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Policy Iteration
    print("\nRunning Policy Iteration...")
    pi_results = []
    for seed in seeds:
        env = OnlineKnapsackEnv()
        env.set_seed(seed)
        
        pi_solver = PolicyIterationOnlineKnapsack(env)
        iterations = pi_solver.run_policy_iteration()
        
        mean_reward, std_reward, knapsack_values = evaluate_policy(env, pi_solver, num_episodes=1, seeds=[seed])
        pi_results.append((mean_reward, std_reward, knapsack_values, pi_solver.V))
        
        print(f"Seed {seed}: {iterations} iterations, Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Plot results
    print("\nGenerating plots...")
    
    # Plot knapsack values for Value Iteration
    for i, (_, _, knapsack_values, _) in enumerate(vi_results):
        plot_knapsack_values(knapsack_values, f"Value Iteration - Seed {seeds[i]}")
    
    # Plot knapsack values for Policy Iteration
    for i, (_, _, knapsack_values, _) in enumerate(pi_results):
        plot_knapsack_values(knapsack_values, f"Policy Iteration - Seed {seeds[i]}")
    
    # Plot value function heatmaps
    plot_value_function_heatmap(vi_results[0][3], "Value Iteration - Value Function Heatmap")
    plot_value_function_heatmap(pi_results[0][3], "Policy Iteration - Value Function Heatmap")
    
    # Test with different step limits
    print("\nTesting with different step limits...")
    step_limits = [10, 50, 500]
    
    for step_limit in step_limits:
        env = OnlineKnapsackEnv()
        env.step_limit = step_limit
        env.set_seed(0)
        
        vi_solver = ValueIterationOnlineKnapsack(env)
        vi_solver.value_iteration()
        
        plot_value_function_heatmap(vi_solver.V, f"Value Iteration - Step Limit {step_limit}")

if __name__ == "__main__":
    main()
