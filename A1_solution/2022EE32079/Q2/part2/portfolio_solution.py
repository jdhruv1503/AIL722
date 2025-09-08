import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'or_gym')))
from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv

class ValueIterationPortfolio:
    def __init__(self, env, gamma=0.999):
        self.env = env
        self.gamma = gamma
        
        self.max_cash = 200
        self.max_price = 20
        self.max_holdings = 10
        
        self.V = np.zeros((self.max_cash + 1, self.max_price + 1, self.max_holdings + 1, env.step_limit + 1))
        
    def get_reward(self, cash, price, holdings, action, step):
        if step == self.env.step_limit - 1:
            return cash + price * holdings
        else:
            return 0
    
    def get_next_state(self, cash, price, holdings, action, step):
        new_cash = cash
        new_holdings = holdings
        
        if action > 0:
            cost = (price + self.env.buy_cost[0]) * action
            if new_cash >= cost and new_holdings + action <= self.env.holding_limit[0]:
                new_cash -= cost
                new_holdings += action
        elif action < 0:
            sell_amount = min(abs(action), holdings)
            new_cash += (price - self.env.sell_cost[0]) * sell_amount
            new_holdings -= sell_amount
        
        new_cash = max(0, min(self.max_cash, new_cash))
        new_holdings = max(0, min(self.max_holdings, new_holdings))
        
        return new_cash, new_holdings
    
    def value_iteration(self, max_iterations=1000):
        for iteration in range(max_iterations):
            delta = 0
            V_old = self.V.copy()
            
            for cash in range(self.max_cash + 1):
                for price in range(self.max_price + 1):
                    for holdings in range(self.max_holdings + 1):
                        for step in range(self.env.step_limit):
                            best_value = float('-inf')
                            
                            for action in [-2, -1, 0, 1, 2]:
                                reward = self.get_reward(cash, price, holdings, action, step)
                                new_cash, new_holdings = self.get_next_state(cash, price, holdings, action, step)
                                
                                if step + 1 < self.env.step_limit:
                                    next_value = self.V[new_cash, price, new_holdings, step + 1]
                                else:
                                    next_value = 0
                                
                                total_value = reward + self.gamma * next_value
                                best_value = max(best_value, total_value)
                            
                            self.V[cash, price, holdings, step] = best_value
                            delta = max(delta, abs(self.V[cash, price, holdings, step] - V_old[cash, price, holdings, step]))
            
            if delta < 1e-6:
                break
        
        return iteration + 1
    
    def get_action(self, state, step):
        cash, price, holdings = int(state[0]), int(state[1]), int(state[2])
        
        cash = max(0, min(self.max_cash, cash))
        price = max(0, min(self.max_price, price))
        holdings = max(0, min(self.max_holdings, holdings))
        
        best_action = 0
        best_value = float('-inf')
        
        for action in [-2, -1, 0, 1, 2]:
            reward = self.get_reward(cash, price, holdings, action, step)
            new_cash, new_holdings = self.get_next_state(cash, price, holdings, action, step)
            
            if step + 1 < self.env.step_limit:
                next_value = self.V[new_cash, price, new_holdings, step + 1]
            else:
                next_value = 0
            
            total_value = reward + self.gamma * next_value
            
            if total_value > best_value:
                best_value = total_value
                best_action = action
        
        return np.array([best_action], dtype=np.int32)

class PolicyIterationPortfolio:
    def __init__(self, env, gamma=0.999, eval_iterations=1000):
        self.env = env
        self.gamma = gamma
        self.eval_iterations = eval_iterations
        
        self.max_cash = 200
        self.max_price = 20
        self.max_holdings = 10
        
        self.policy = np.zeros((self.max_cash + 1, self.max_price + 1, self.max_holdings + 1, self.env.step_limit + 1), dtype=int)
        self.V = np.zeros((self.max_cash + 1, self.max_price + 1, self.max_holdings + 1, self.env.step_limit + 1))
        
    def get_reward(self, cash, price, holdings, action, step):
        if step == self.env.step_limit - 1:
            return cash + price * holdings
        else:
            return 0
    
    def get_next_state(self, cash, price, holdings, action, step):
        new_cash = cash
        new_holdings = holdings
        
        if action > 0:
            cost = (price + self.env.buy_cost[0]) * action
            if new_cash >= cost and new_holdings + action <= self.env.holding_limit[0]:
                new_cash -= cost
                new_holdings += action
        elif action < 0:
            sell_amount = min(abs(action), holdings)
            new_cash += (price - self.env.sell_cost[0]) * sell_amount
            new_holdings -= sell_amount
        
        new_cash = max(0, min(self.max_cash, new_cash))
        new_holdings = max(0, min(self.max_holdings, new_holdings))
        
        return new_cash, new_holdings
    
    def policy_evaluation(self):
        for _ in range(self.eval_iterations):
            delta = 0
            V_old = self.V.copy()
            
            for cash in range(self.max_cash + 1):
                for price in range(self.max_price + 1):
                    for holdings in range(self.max_holdings + 1):
                        for step in range(self.env.step_limit):
                            action = self.policy[cash, price, holdings, step]
                            reward = self.get_reward(cash, price, holdings, action, step)
                            new_cash, new_holdings = self.get_next_state(cash, price, holdings, action, step)
                            
                            if step + 1 < self.env.step_limit:
                                next_value = self.V[new_cash, price, new_holdings, step + 1]
                            else:
                                next_value = 0
                            
                            self.V[cash, price, holdings, step] = reward + self.gamma * next_value
                            delta = max(delta, abs(self.V[cash, price, holdings, step] - V_old[cash, price, holdings, step]))
            
            if delta < 1e-6:
                break
    
    def policy_improvement(self):
        policy_stable = True
        
        for cash in range(self.max_cash + 1):
            for price in range(self.max_price + 1):
                for holdings in range(self.max_holdings + 1):
                    for step in range(self.env.step_limit):
                        old_action = self.policy[cash, price, holdings, step]
                        
                        best_action = 0
                        best_value = float('-inf')
                        
                        for action in [-2, -1, 0, 1, 2]:
                            reward = self.get_reward(cash, price, holdings, action, step)
                            new_cash, new_holdings = self.get_next_state(cash, price, holdings, action, step)
                            
                            if step + 1 < self.env.step_limit:
                                next_value = self.V[new_cash, price, new_holdings, step + 1]
                            else:
                                next_value = 0
                            
                            total_value = reward + self.gamma * next_value
                            
                            if total_value > best_value:
                                best_value = total_value
                                best_action = action
                        
                        self.policy[cash, price, holdings, step] = best_action
                        
                        if best_action != old_action:
                            policy_stable = False
        
        return policy_stable
    
    def run_policy_iteration(self, max_iterations=1000):
        for iteration in range(max_iterations):
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            
            if policy_stable:
                break
        
        return iteration + 1
    
    def get_action(self, state, step):
        cash, price, holdings = int(state[0]), int(state[1]), int(state[2])
        
        cash = max(0, min(self.max_cash, cash))
        price = max(0, min(self.max_price, price))
        holdings = max(0, min(self.max_holdings, holdings))
        
        action = self.policy[cash, price, holdings, step]
        return np.array([action], dtype=np.int32)

def evaluate_policy(env, policy, num_episodes=20):
    rewards = []
    wealth_evolution = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        episode_wealth = []
        done = False
        step = 0
        
        while not done and step < env.step_limit:
            action = policy.get_action(state, step)
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            current_wealth = state[0] + state[1] * state[2]
            episode_wealth.append(current_wealth)
            
            step += 1
        
        rewards.append(total_reward)
        wealth_evolution.append(episode_wealth)
    
    return np.mean(rewards), np.std(rewards), wealth_evolution

def plot_wealth_evolution(wealth_evolution, title="Wealth Evolution"):
    plt.figure(figsize=(12, 8))
    
    for i, wealth in enumerate(wealth_evolution):
        plt.plot(wealth, label=f'Episode {i+1}', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Total Wealth')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')

def plot_training_progress(rewards_history, title="Training Progress"):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Portfolio Wealth')
    plt.grid(True)
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')

def main():
    print("Portfolio Optimization Analysis")
    
    price_sequences = [
        [1, 3, 5, 5, 4, 3, 2, 3, 5, 8],
        [2, 2, 2, 4, 2, 2, 4, 2, 2, 2],
        [4, 1, 4, 1, 4, 4, 4, 1, 1, 4]
    ]
    
    gammas = [0.999, 1.0]
    
    for gamma in gammas:
        print(f"\nGamma = {gamma}")
        
        for i, prices in enumerate(price_sequences):
            print(f"\nPrice Sequence {i+1}: {prices}")
            
            env = DiscretePortfolioOptEnv(prices=prices)
            
            print("Running Value Iteration...")
            start_time = time.time()
            vi_solver = ValueIterationPortfolio(env, gamma=gamma)
            iterations = vi_solver.value_iteration()
            vi_time = time.time() - start_time
            
            vi_mean, vi_std, vi_wealth = evaluate_policy(env, vi_solver)
            print(f"VI: {iterations} iterations, {vi_time:.2f}s, Mean wealth: {vi_mean:.2f} ± {vi_std:.2f}")
            
            print("Running Policy Iteration...")
            start_time = time.time()
            pi_solver = PolicyIterationPortfolio(env, gamma=gamma)
            iterations = pi_solver.run_policy_iteration()
            pi_time = time.time() - start_time
            
            pi_mean, pi_std, pi_wealth = evaluate_policy(env, pi_solver)
            print(f"PI: {iterations} iterations, {pi_time:.2f}s, Mean wealth: {pi_mean:.2f} ± {pi_std:.2f}")
            
            plot_wealth_evolution(vi_wealth, f"Value Iteration - Gamma {gamma} - Sequence {i+1}")
            plot_wealth_evolution(pi_wealth, f"Policy Iteration - Gamma {gamma} - Sequence {i+1}")
    
    print("\nTesting with variance = 1.0")
    env = DiscretePortfolioOptEnv(variance=1.0)
    
    print("Running Policy Iteration with variance...")
    pi_solver = PolicyIterationPortfolio(env, gamma=0.999)
    
    convergence_history = []
    for iteration in range(1000):
        old_policy = pi_solver.policy.copy()
        pi_solver.policy_evaluation()
        pi_solver.policy_improvement()
        
        policy_diff = np.max(np.abs(pi_solver.policy - old_policy))
        convergence_history.append(policy_diff)
        
        if policy_diff < 1e-2:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_history)
    plt.title("Policy Iteration Convergence with Variance")
    plt.xlabel('Iteration')
    plt.ylabel('Maximum Policy Difference')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('policy_iteration_convergence_with_variance.png', dpi=300, bbox_inches='tight')
    
    with open('q2_part2_results.txt', 'w') as f:
        f.write("Q2 Part 2 Results\n")
        f.write("================\n\n")
        f.write("Portfolio Optimization Results:\n")
        f.write("Gamma 0.999:\n")
        f.write("- Value Iteration: Various iterations, execution times\n")
        f.write("- Policy Iteration: Various iterations, execution times\n")
        f.write("Gamma 1.0:\n")
        f.write("- Value Iteration: Various iterations, execution times\n")
        f.write("- Policy Iteration: Various iterations, execution times\n")
        f.write("Variance 1.0:\n")
        f.write("- Policy Iteration convergence analysis\n")

if __name__ == "__main__":
    main()