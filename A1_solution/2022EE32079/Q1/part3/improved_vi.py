import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env import FootballSkillsEnv
from collections import deque

def prioritized_value_iteration(envr=FootballSkillsEnv, gamma=0.95, theta=1e-6):
    env = envr(render_mode='gif')
    num_states = env.grid_size * env.grid_size * 2
    num_actions = env.action_space.n
    
    value_function = np.zeros(num_states)
    transition_calls = 0
    iteration = 0
    
    priority_queue = deque()
    
    for s in range(num_states):
        if not env._is_terminal(env.index_to_state(s)):
            priority_queue.append((float('inf'), s))
    
    while priority_queue:
        iteration += 1
        
        priority_queue = deque(sorted(priority_queue, key=lambda x: x[0], reverse=True))
        
        if not priority_queue:
            break
            
        priority, s = priority_queue.popleft()
        
        if env._is_terminal(env.index_to_state(s)):
            continue
        
        old_value = value_function[s]
        
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
            
            best_value = max(best_value, expected_value)
        
        value_function[s] = best_value
        
        bellman_error = abs(value_function[s] - old_value)
        
        if bellman_error > theta:
            priority_queue.append((bellman_error, s))
    
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

def standard_value_iteration(envr=FootballSkillsEnv, gamma=0.95, theta=1e-6):
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

def policy_iteration(envr=FootballSkillsEnv, gamma=0.95, theta=1e-6):
    env = envr(render_mode='gif')
    num_states = env.grid_size * env.grid_size * 2
    num_actions = env.action_space.n
    
    policy = np.random.randint(0, num_actions, num_states)
    value_function = np.zeros(num_states)
    
    transition_calls = 0
    iteration = 0
    
    while True:
        iteration += 1
        
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

def evaluate_policy(env, policy, num_episodes=20):
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
    print("Improved Value Iteration Analysis")
    
    pvi_policy, pvi_values, pvi_iter, pvi_calls = prioritized_value_iteration()
    pvi_mean, pvi_std = evaluate_policy(FootballSkillsEnv(), pvi_policy)
    
    svi_policy, svi_values, svi_iter, svi_calls = standard_value_iteration()
    svi_mean, svi_std = evaluate_policy(FootballSkillsEnv(), svi_policy)
    
    pi_policy, pi_values, pi_iter, pi_calls = policy_iteration()
    pi_mean, pi_std = evaluate_policy(FootballSkillsEnv(), pi_policy)
    
    print(f"Prioritized VI: {pvi_iter} iter, {pvi_calls} calls, {pvi_mean:.2f}±{pvi_std:.2f}")
    print(f"Standard VI: {svi_iter} iter, {svi_calls} calls, {svi_mean:.2f}±{svi_std:.2f}")
    print(f"Policy Iteration: {pi_iter} iter, {pi_calls} calls, {pi_mean:.2f}±{pi_std:.2f}")
    
    policies_identical_pvi_svi = np.array_equal(pvi_policy, svi_policy)
    policies_identical_pvi_pi = np.array_equal(pvi_policy, pi_policy)
    policies_identical_svi_pi = np.array_equal(svi_policy, pi_policy)
    
    print(f"Prioritized VI vs Standard VI identical: {policies_identical_pvi_svi}")
    print(f"Prioritized VI vs Policy Iteration identical: {policies_identical_pvi_pi}")
    print(f"Standard VI vs Policy Iteration identical: {policies_identical_svi_pi}")
    
    env = FootballSkillsEnv(render_mode='gif')
    env.get_gif(pvi_policy, seed=20, filename="prioritized_vi.gif")
    
    plt.figure(figsize=(12, 8))
    algorithms = ['Prioritized VI', 'Standard VI', 'Policy Iteration']
    calls = [pvi_calls, svi_calls, pi_calls]
    rewards = [pvi_mean, svi_mean, pi_mean]
    iters = [pvi_iter, svi_iter, pi_iter]
    
    plt.subplot(2, 2, 1)
    plt.bar(algorithms, calls)
    plt.title('Transition Calls Comparison')
    plt.ylabel('Number of Calls')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    plt.bar(algorithms, rewards)
    plt.title('Performance Comparison')
    plt.ylabel('Mean Reward')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    plt.bar(algorithms, iters)
    plt.title('Iterations Comparison')
    plt.ylabel('Number of Iterations')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('q1_part3_results.png', dpi=300, bbox_inches='tight')
    
    with open('q1_part3_results.txt', 'w') as f:
        f.write("Q1 Part 3 Results\n")
        f.write("================\n\n")
        f.write(f"Prioritized VI: {pvi_iter} iterations, {pvi_calls} calls, {pvi_mean:.2f}±{pvi_std:.2f}\n")
        f.write(f"Standard VI: {svi_iter} iterations, {svi_calls} calls, {svi_mean:.2f}±{svi_std:.2f}\n")
        f.write(f"Policy Iteration: {pi_iter} iterations, {pi_calls} calls, {pi_mean:.2f}±{pi_std:.2f}\n")
        f.write(f"Prioritized VI vs Standard VI identical: {policies_identical_pvi_svi}\n")
        f.write(f"Prioritized VI vs Policy Iteration identical: {policies_identical_pvi_pi}\n")
        f.write(f"Standard VI vs Policy Iteration identical: {policies_identical_svi_pi}\n")

if __name__ == "__main__":
    main()