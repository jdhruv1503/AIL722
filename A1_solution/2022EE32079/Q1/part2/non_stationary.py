import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env import FootballSkillsEnv

def time_dependent_value_iteration(envr=FootballSkillsEnv, gamma=0.95, max_horizon=40):
    env = envr(render_mode='gif', degrade_pitch=True)
    num_states = env.grid_size * env.grid_size * 2
    num_actions = env.action_space.n
    
    V = np.zeros((max_horizon, num_states))
    policy = np.zeros((max_horizon, num_states), dtype=int)
    transition_calls = 0
    
    for t in range(max_horizon - 1, -1, -1):
        for s in range(num_states):
            if env._is_terminal(env.index_to_state(s)):
                continue
            
            best_value = float('-inf')
            best_action = None
            
            for a in range(num_actions):
                transitions = env.get_transitions_at_time(env.index_to_state(s), a, t)
                transition_calls += 1
                
                if not transitions:
                    continue
                
                expected_value = 0
                for prob, next_state in transitions:
                    next_s = env.state_to_index(next_state)
                    reward = env._get_reward(next_state[:2], a, env.index_to_state(s)[:2])
                    
                    if t + 1 < max_horizon:
                        expected_value += prob * (reward + gamma * V[t + 1, next_s])
                    else:
                        expected_value += prob * reward
                
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = a
            
            V[t, s] = best_value
            policy[t, s] = best_action
    
    return policy, transition_calls

def stationary_value_iteration_degraded(envr=FootballSkillsEnv, gamma=0.95, theta=1e-6):
    env = envr(render_mode='gif', degrade_pitch=True)
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

def evaluate_time_dependent_policy(env, policy, num_episodes=20):
    rewards = []
    
    for seed in range(num_episodes):
        obs, _ = env.reset(seed=seed)
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            s_index = env.state_to_index(obs)
            time_step = min(steps, len(policy) - 1)
            action = policy[time_step][s_index]
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
        
        rewards.append(total_reward)
    
    return np.mean(rewards), np.std(rewards)

def evaluate_stationary_policy_degraded(env, policy, num_episodes=20):
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
    print("Non-Stationary Environment Analysis")
    
    td_policy, td_calls = time_dependent_value_iteration()
    td_mean, td_std = evaluate_time_dependent_policy(FootballSkillsEnv(render_mode='gif', degrade_pitch=True), td_policy)
    
    stat_policy, stat_values, stat_iter, stat_calls = stationary_value_iteration_degraded()
    stat_mean, stat_std = evaluate_stationary_policy_degraded(FootballSkillsEnv(render_mode='gif', degrade_pitch=True), stat_policy)
    
    print(f"Time-dependent VI: {td_calls} calls, {td_mean:.2f}±{td_std:.2f}")
    print(f"Stationary VI: {stat_iter} iter, {stat_calls} calls, {stat_mean:.2f}±{stat_std:.2f}")
    
    print("Generating GIFs...")
    env = FootballSkillsEnv(render_mode='gif', degrade_pitch=True)
    env.get_gif(td_policy, seed=20, filename="time_dependent.gif")
    
    plt.figure(figsize=(10, 6))
    algorithms = ['Time-dependent VI', 'Stationary VI']
    calls = [td_calls, stat_calls]
    rewards = [td_mean, stat_mean]
    
    plt.subplot(1, 2, 1)
    plt.bar(algorithms, calls)
    plt.title('Transition Calls Comparison')
    plt.ylabel('Number of Calls')
    
    plt.subplot(1, 2, 2)
    plt.bar(algorithms, rewards)
    plt.title('Performance Comparison')
    plt.ylabel('Mean Reward')
    
    plt.tight_layout()
    plt.savefig('q1_part2_results.png', dpi=300, bbox_inches='tight')
    
    with open('q1_part2_results.txt', 'w') as f:
        f.write("Q1 Part 2 Results\n")
        f.write("================\n\n")
        f.write(f"Time-dependent VI: {td_calls} calls, {td_mean:.2f}±{td_std:.2f}\n")
        f.write(f"Stationary VI: {stat_iter} iterations, {stat_calls} calls, {stat_mean:.2f}±{stat_std:.2f}\n")

if __name__ == "__main__":
    main()