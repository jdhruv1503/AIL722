import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env import FootballSkillsEnv

def policy_iteration(envr=FootballSkillsEnv, gamma=0.95, theta=1e-6):
    '''
    Implements the Policy Iteration algorithm to find the optimal policy for the 
    Football Skills Environment.
    
    Args:
        envr (class, optional): Environment class to instantiate. Defaults to FootballSkillsEnv.
        gamma (float): Discount factor
        theta (float): Convergence threshold
    
    Returns:
        tuple: (optimal_policy, value_function, num_iterations, transition_calls)
            - optimal_policy (dict): Maps state indices to optimal actions
            - value_function (numpy.ndarray): Value of each state under optimal policy  
            - num_iterations (int): Number of iterations until convergence
            - transition_calls (int): Total number of calls to get_transitions_at_time
    '''
    env = envr(render_mode='gif')
    num_states = env.grid_size * env.grid_size * 2
    num_actions = env.action_space.n
    
    # Initialize policy randomly
    policy = np.random.randint(0, num_actions, num_states)
    value_function = np.zeros(num_states)
    
    transition_calls = 0
    iteration = 0
    
    while True:
        iteration += 1
        
        # Policy Evaluation
        while True:
            delta = 0
            old_values = value_function.copy()
            
            for s in range(num_states):
                if env._is_terminal(env.index_to_state(s)):
                    continue
                    
                action = policy[s]
                transitions = env.get_transitions_at_time(env.index_to_state(s), action)
                transition_calls += 1
                
                if not transitions:
                    continue
                
                expected_value = 0
                for prob, next_state in transitions:
                    next_s = env.state_to_index(next_state)
                    reward = env._get_reward(next_state[:2], action, env.index_to_state(s)[:2])
                    expected_value += prob * (reward + gamma * old_values[next_s])
                
                value_function[s] = expected_value
                delta = max(delta, abs(value_function[s] - old_values[s]))
            
            if delta < theta:
                break
        
        # Policy Improvement
        policy_stable = True
        for s in range(num_states):
            if env._is_terminal(env.index_to_state(s)):
                continue
                
            old_action = policy[s]
            best_action = None
            best_value = float('-inf')
            
            for a in range(num_actions):
                transitions = env.get_transitions_at_time(env.index_to_state(s), a)
                transition_calls += 1
                
                if not transitions:
                    continue
                
                expected_value = 0
                for prob, next_state in transitions:
                    next_s = env.state_to_index(next_state)
                    reward = env._get_reward(next_state[:2], a, env.index_to_state(s)[:2])
                    expected_value += prob * (reward + gamma * value_function[next_s])
                
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = a
            
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False
        
        if policy_stable:
            break
    
    return policy, value_function, iteration, transition_calls

def value_iteration(envr=FootballSkillsEnv, gamma=0.95, theta=1e-6):
    '''
    Implements the Value Iteration algorithm to find the optimal policy for the 
    Football Skills Environment.
    
    Args:
        envr (class, optional): Environment class to instantiate. Defaults to FootballSkillsEnv.
        gamma (float): Discount factor
        theta (float): Convergence threshold
    
    Returns:
        tuple: (optimal_policy, value_function, num_iterations, transition_calls)
            - optimal_policy (dict): Maps state indices to optimal actions
            - value_function (numpy.ndarray): Value of each state under optimal policy  
            - num_iterations (int): Number of iterations until convergence
            - transition_calls (int): Total number of calls to get_transitions_at_time
    '''
    env = envr(render_mode='gif')
    num_states = env.grid_size * env.grid_size * 2
    num_actions = env.action_space.n
    
    value_function = np.zeros(num_states)
    transition_calls = 0
    iteration = 0
    
    while True:
        iteration += 1
        delta = 0
        old_values = value_function.copy()
        
        for s in range(num_states):
            if env._is_terminal(env.index_to_state(s)):
                continue
            
            best_value = float('-inf')
            for a in range(num_actions):
                transitions = env.get_transitions_at_time(env.index_to_state(s), a)
                transition_calls += 1
                
                if not transitions:
                    continue
                
                expected_value = 0
                for prob, next_state in transitions:
                    next_s = env.state_to_index(next_state)
                    reward = env._get_reward(next_state[:2], a, env.index_to_state(s)[:2])
                    expected_value += prob * (reward + gamma * old_values[next_s])
                
                best_value = max(best_value, expected_value)
            
            value_function[s] = best_value
            delta = max(delta, abs(value_function[s] - old_values[s]))
        
        if delta < theta:
            break
    
    # Extract policy
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        if env._is_terminal(env.index_to_state(s)):
            continue
        
        best_action = None
        best_value = float('-inf')
        
        for a in range(num_actions):
            transitions = env.get_transitions_at_time(env.index_to_state(s), a)
            transition_calls += 1
            
            if not transitions:
                continue
            
            expected_value = 0
            for prob, next_state in transitions:
                next_s = env.state_to_index(next_state)
                reward = env._get_reward(next_state[:2], a, env.index_to_state(s)[:2])
                expected_value += prob * (reward + gamma * value_function[next_s])
            
            if expected_value > best_value:
                best_value = expected_value
                best_action = a
        
        policy[s] = best_action
    
    return policy, value_function, iteration, transition_calls

def evaluate_policy(env, policy, num_episodes=20):
    '''Evaluate a policy by running multiple episodes'''
    rewards = []
    
    for seed in range(num_episodes):
        obs, _ = env.reset(seed=seed)
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            s_index = env.state_to_index(obs)
            action = policy[s_index]
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
        
        rewards.append(total_reward)
    
    return np.mean(rewards), np.std(rewards)

def main():
    # Test with different discount factors
    gammas = [0.95, 0.5, 0.3]
    
    for gamma in gammas:
        print(f"\n=== Testing with gamma = {gamma} ===")
        
        # Policy Iteration
        print("Running Policy Iteration...")
        pi_policy, pi_values, pi_iter, pi_calls = policy_iteration(gamma=gamma)
        pi_mean, pi_std = evaluate_policy(FootballSkillsEnv(), pi_policy)
        
        # Value Iteration  
        print("Running Value Iteration...")
        vi_policy, vi_values, vi_iter, vi_calls = value_iteration(gamma=gamma)
        vi_mean, vi_std = evaluate_policy(FootballSkillsEnv(), vi_policy)
        
        print(f"Policy Iteration: {pi_iter} iterations, {pi_calls} transition calls")
        print(f"Policy Iteration Performance: {pi_mean:.2f} ± {pi_std:.2f}")
        
        print(f"Value Iteration: {vi_iter} iterations, {vi_calls} transition calls")
        print(f"Value Iteration Performance: {vi_mean:.2f} ± {vi_std:.2f}")
        
        # Check if policies are identical
        policies_identical = np.array_equal(pi_policy, vi_policy)
        print(f"Policies identical: {policies_identical}")
    
    # Generate GIFs for gamma=0.95
    print("\nGenerating GIFs...")
    env = FootballSkillsEnv(render_mode='gif')
    
    pi_policy, _, _, _ = policy_iteration(gamma=0.95)
    vi_policy, _, _, _ = value_iteration(gamma=0.95)
    
    env.get_gif(pi_policy, seed=20, filename="policy_iteration.gif")
    env.get_gif(vi_policy, seed=20, filename="value_iteration.gif")

if __name__ == "__main__":
    main()
