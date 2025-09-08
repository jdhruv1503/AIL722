import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'or_gym')))
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv

class ValueIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-4):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.max_weight = env.max_weight
        self.num_items = env.N
        self.max_item_weight = 100
        self.max_item_value = 100
        
        self.V = np.zeros((self.max_weight + 1, self.num_items, self.max_item_weight + 1, self.max_item_value + 1))
        
    def get_reward_and_done(self, current_weight, item_idx, action):
        if action == 0:
            return 0, False
        else:
            item_weight = self.env.item_weights[item_idx]
            if current_weight + item_weight <= self.max_weight:
                return self.env.item_values[item_idx], False
            else:
                return 0, True
    
    def value_iteration(self, max_iterations=1000):
        for iteration in range(max_iterations):
            delta = 0
            V_old = self.V.copy()
            
            for w in range(self.max_weight + 1):
                for item_idx in range(self.num_items):
                    item_weight = self.env.item_weights[item_idx]
                    item_value = self.env.item_values[item_idx]
                    
                    if item_weight > self.max_weight - w:
                        continue
                    
                    reject_reward, reject_done = self.get_reward_and_done(w, item_idx, 0)
                    if not reject_done:
                        reject_value = reject_reward + self.gamma * V_old[w, (item_idx + 1) % self.num_items, item_weight, item_value]
                    else:
                        reject_value = reject_reward
                    
                    accept_reward, accept_done = self.get_reward_and_done(w, item_idx, 1)
                    if not accept_done:
                        new_weight = w + item_weight
                        accept_value = accept_reward + self.gamma * V_old[new_weight, (item_idx + 1) % self.num_items, item_weight, item_value]
                    else:
                        accept_value = accept_reward
                    
                    self.V[w, item_idx, item_weight, item_value] = max(reject_value, accept_value)
                    delta = max(delta, abs(self.V[w, item_idx, item_weight, item_value] - V_old[w, item_idx, item_weight, item_value]))
            
            if delta < self.epsilon:
                break
        
        return iteration + 1
    
    def get_action(self, state):
        if isinstance(state, dict):
            state = state['state']
        
        current_weight, item_idx, item_weight, item_value = state
        
        reject_reward, reject_done = self.get_reward_and_done(current_weight, item_idx, 0)
        if not reject_done:
            reject_value = reject_reward + self.gamma * self.V[current_weight, (item_idx + 1) % self.num_items, item_weight, item_value]
        else:
            reject_value = reject_reward
        
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
        
        self.max_weight = env.max_weight
        self.num_items = env.N
        self.max_item_weight = 100
        self.max_item_value = 100
        
        self.policy = np.zeros((self.max_weight + 1, self.num_items, self.max_item_weight + 1, self.max_item_value + 1), dtype=int)
        self.V = np.zeros((self.max_weight + 1, self.num_items, self.max_item_weight + 1, self.max_item_value + 1))
        
    def get_reward_and_done(self, current_weight, item_idx, action):
        if action == 0:
            return 0, False
        else:
            item_weight = self.env.item_weights[item_idx]
            if current_weight + item_weight <= self.max_weight:
                return self.env.item_values[item_idx], False
            else:
                return 0, True
    
    def policy_evaluation(self):
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
                        if action == 0:
                            next_value = self.V[w, (item_idx + 1) % self.num_items, item_weight, item_value]
                        else:
                            new_weight = w + item_weight
                            next_value = self.V[new_weight, (item_idx + 1) % self.num_items, item_weight, item_value]
                        
                        self.V[w, item_idx, item_weight, item_value] = reward + self.gamma * next_value
                    else:
                        self.V[w, item_idx, item_weight, item_value] = reward
                    
                    delta = max(delta, abs(self.V[w, item_idx, item_weight, item_value] - V_old[w, item_idx, item_weight, item_value]))
            
            if delta < self.epsilon:
                break
    
    def policy_improvement(self):
        policy_stable = True
        
        for w in range(self.max_weight + 1):
            for item_idx in range(self.num_items):
                item_weight = self.env.item_weights[item_idx]
                item_value = self.env.item_values[item_idx]
                
                if item_weight > self.max_weight - w:
                    continue
                
                old_action = self.policy[w, item_idx, item_weight, item_value]
                
                reject_reward, reject_done = self.get_reward_and_done(w, item_idx, 0)
                if not reject_done:
                    reject_value = reject_reward + self.gamma * self.V[w, (item_idx + 1) % self.num_items, item_weight, item_value]
                else:
                    reject_value = reject_reward
                
                accept_reward, accept_done = self.get_reward_and_done(w, item_idx, 1)
                if not accept_done:
                    new_weight = w + item_weight
                    accept_value = accept_reward + self.gamma * self.V[new_weight, (item_idx + 1) % self.num_items, item_weight, item_value]
                else:
                    accept_value = accept_reward
                
                self.policy[w, item_idx, item_weight, item_value] = 1 if accept_value > reject_value else 0
                
                if self.policy[w, item_idx, item_weight, item_value] != old_action:
                    policy_stable = False
        
        return policy_stable
    
    def run_policy_iteration(self, max_iterations=1000):
        for iteration in range(max_iterations):
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            
            if policy_stable:
                break
        
        return iteration + 1
    
    def get_action(self, state):
        if isinstance(state, dict):
            state = state['state']
        
        current_weight, item_idx, item_weight, item_value = state
        return self.policy[current_weight, item_idx, item_weight, item_value]

def evaluate_policy(env, policy, num_episodes=5, seeds=None):
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
        knapsack_values.append(total_reward)
    
    return np.mean(rewards), np.std(rewards), knapsack_values

def plot_knapsack_values(knapsack_values, title="Knapsack Values"):
    plt.figure(figsize=(10, 6))
    plt.plot(knapsack_values, marker='o')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Knapsack Value')
    plt.grid(True)
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')

def plot_value_function_heatmap(V, title="Value Function Heatmap"):
    weights = np.arange(V.shape[0])
    values = np.arange(V.shape[3])
    
    V_2d = np.mean(V, axis=(1, 2))
    
    plt.figure(figsize=(12, 8))
    plt.imshow(V_2d, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('Item Value')
    plt.ylabel('Current Weight')
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')

def main():
    print("Online Knapsack Problem Analysis")
    
    seeds = [0, 1, 2, 3, 4]
    
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
    
    for i, (_, _, knapsack_values, _) in enumerate(vi_results):
        plot_knapsack_values(knapsack_values, f"Value Iteration - Seed {seeds[i]}")
    
    for i, (_, _, knapsack_values, _) in enumerate(pi_results):
        plot_knapsack_values(knapsack_values, f"Policy Iteration - Seed {seeds[i]}")
    
    plot_value_function_heatmap(vi_results[0][3], "Value Iteration - Value Function Heatmap")
    plot_value_function_heatmap(pi_results[0][3], "Policy Iteration - Value Function Heatmap")
    
    step_limits = [10, 50, 500]
    
    for step_limit in step_limits:
        env = OnlineKnapsackEnv()
        env.step_limit = step_limit
        env.set_seed(0)
        
        vi_solver = ValueIterationOnlineKnapsack(env)
        vi_solver.value_iteration()
        
        plot_value_function_heatmap(vi_solver.V, f"Value Iteration - Step Limit {step_limit}")
    
    with open('q2_part1_results.txt', 'w') as f:
        f.write("Q2 Part 1 Results\n")
        f.write("================\n\n")
        f.write("Value Iteration Results:\n")
        for i, (mean, std, _, _) in enumerate(vi_results):
            f.write(f"Seed {seeds[i]}: {mean:.2f}±{std:.2f}\n")
        f.write("\nPolicy Iteration Results:\n")
        for i, (mean, std, _, _) in enumerate(pi_results):
            f.write(f"Seed {seeds[i]}: {mean:.2f}±{std:.2f}\n")

if __name__ == "__main__":
    main()