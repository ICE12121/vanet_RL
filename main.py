import numpy as np
import matplotlib.pyplot as plt
from vanet_env.environment import VANETEnvironment
from rl_agent.dqn_agent import DQNAgent
from visualization.performance_metrics import PerformanceMetrics
import torch

# =============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE TO CUSTOMIZE YOUR SIMULATION
# =============================================================================

# Environment Configuration
NUM_VEHICLES = 15           # Number of vehicles in the simulation
NUM_RSUS = 9               # Number of RSUs (Road Side Units) 
AREA_SIZE = 1000           # Size of the simulation area (1000x1000 meters)
RSU_SPACING = 2            # RSUs placed every N intersections (2 = every 2 blocks)

# Training Configuration  
EPISODES = 1000            # Number of training episodes
BATCH_SIZE = 32            # Batch size for training
TARGET_UPDATE_FREQ = 10    # How often to update target network
RENDER = False

# Agent Configuration
LEARNING_RATE = 0.001      # Learning rate for the neural network
EPSILON_START = 1.0        # Starting exploration rate
EPSILON_END = 0.01         # Minimum exploration rate
EPSILON_DECAY = 0.995      # Exploration decay rate

# =============================================================================
# TRAINING AND SIMULATION CODE
# =============================================================================

def train_agent(env, agent, episodes, batch_size=32, target_update_freq=10):
    rewards_history = []
    avg_rewards_history = []
    performance_metrics = PerformanceMetrics()
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            
            # Update performance metrics
            performance_metrics.update(env, action, reward, step)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1
            
            agent.replay(batch_size)
        
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        rewards_history.append(total_reward)
        avg_rewards_history.append(np.mean(rewards_history[-100:]))
        
        if episode % 10 == 0:
            print(f"Episode: {episode}, Average Reward: {avg_rewards_history[-1]:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    # Plot performance metrics at the end of training
    performance_metrics.plot_metrics()
    
    return rewards_history, avg_rewards_history

def plot_results(rewards_history, avg_rewards_history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(avg_rewards_history)
    plt.title('Average Rewards (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

def main():
    print("🚗📡 VANET RL-Based Handover Decision Making Simulation")
    print("="*60)
    print(f"Configuration:")
    print(f"  🚗 Vehicles: {NUM_VEHICLES}")
    print(f"  📡 RSUs: {NUM_RSUS}")
    print(f"  🗺️  Area: {AREA_SIZE}×{AREA_SIZE}m")
    print(f"  🧠 Episodes: {EPISODES}")
    print(f"  📊 Batch Size: {BATCH_SIZE}")
    print("="*60)
    
    # Initialize environment
    env = VANETEnvironment(
        num_vehicles=NUM_VEHICLES,
        num_rsus=NUM_RSUS,
        area_size=AREA_SIZE
    )
    
    # Get state and action dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"🔧 Environment Info:")
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")
    print(f"  RSU Coverage: 250m radius")
    print(f"  RSU Capacity: 10 vehicles each")
    print("="*60)
    
    # Initialize agent
    agent = DQNAgent(state_size, action_size)
    
    # Initialize performance tracking
    metrics = PerformanceMetrics()
    
    # Training loop
    scores = []
    avg_scores = []
    epsilon_values = []
    
    print("🚀 Starting Training...")
    print("Episode | Score | Avg Score | Epsilon | Handovers | Success Rate")
    print("-" * 70)
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_score = 0
        step = 0
        
        while True:
            # Choose action
            action = agent.act(state)
            
            # Execute action
            next_state, reward, done, _, _ = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update metrics
            metrics.update(env, action, reward, step)
            
            state = next_state
            total_score += reward
            step += 1
            
            if done:
                break
        
        # End episode tracking
        metrics.end_episode()
        
        # Train agent
        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)
        
        # Update target network
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Track performance
        scores.append(total_score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        epsilon_values.append(agent.epsilon)
        
        # Print progress
        if episode % 50 == 0 or episode == EPISODES - 1:
            episode_stats = metrics.episode_metrics[-1] if metrics.episode_metrics else {}
            success_rate = episode_stats.get('handover_success_rate', 0)
            handover_count = episode_stats.get('total_handovers', 0)
            
            print(f"{episode:7d} | {total_score:5.0f} | {avg_score:9.0f} | {agent.epsilon:.3f} | {handover_count:9.0f} | {success_rate:11.1%}")
    
    print("="*70)
    print("✅ Training Completed!")
    
    # Save trained model
    agent.save('trained_agent.pth')
    print("💾 Model saved as 'trained_agent.pth'")
    
    # Test baseline strategies for comparison
    print("\n🔄 Testing Baseline Strategies for Comparison...")
    baseline_results = metrics.compare_with_baselines(env, episodes=5)
    
    # Generate comprehensive performance plots
    print("📊 Generating Performance Plots...")
    
    # 1. Training progress plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(scores, alpha=0.6, label='Episode Score')
    plt.plot(avg_scores, 'r-', linewidth=2, label='100-Episode Average')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Score Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(epsilon_values)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon (Exploration Rate)')
    plt.title('Exploration vs Exploitation')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    if metrics.episode_metrics:
        episodes_range = range(1, len(metrics.episode_metrics) + 1)
        success_rates = [e['handover_success_rate'] for e in metrics.episode_metrics]
        plt.plot(episodes_range, success_rates, 'g-', alpha=0.6, label='Success Rate')
        
        # Add moving average
        if len(success_rates) > 10:
            window = min(20, len(success_rates) // 5)
            moving_avg = np.convolve(success_rates, np.ones(window)/window, mode='valid')
            plt.plot(episodes_range[window-1:], moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Average')
        
        plt.xlabel('Episode')
        plt.ylabel('Handover Success Rate')
        plt.title('Handover Decision Quality')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    if metrics.episode_metrics:
        efficiency_scores = [e['system_efficiency'] for e in metrics.episode_metrics]
        plt.plot(episodes_range, efficiency_scores, 'purple', alpha=0.6, label='System Efficiency')
        
        # Add moving average
        if len(efficiency_scores) > 10:
            window = min(20, len(efficiency_scores) // 5)
            moving_avg = np.convolve(efficiency_scores, np.ones(window)/window, mode='valid')
            plt.plot(episodes_range[window-1:], moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Average')
        
        plt.xlabel('Episode')
        plt.ylabel('Efficiency Score')
        plt.title('Overall System Efficiency')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed performance metrics
    metrics.plot_metrics('detailed_performance_metrics.png')
    
    # 3. Baseline comparison
    metrics.plot_comparison_with_baselines('baseline_comparison.png')
    
    # 4. Learning progress
    metrics.plot_learning_progress('learning_progress.png')
    
    # Print final results summary
    print("\n" + "="*60)
    print("🎯 FINAL TRAINING RESULTS")
    print("="*60)
    print(f"📈 Final Average Score: {avg_scores[-1]:,.0f}")
    print(f"📊 Best 100-Episode Average: {max(avg_scores):,.0f}")
    print(f"🎲 Final Exploration Rate: {agent.epsilon:.1%}")
    
    if metrics.episode_metrics:
        final_episode = metrics.episode_metrics[-1]
        print(f"✅ Final Handover Success Rate: {final_episode['handover_success_rate']:.1%}")
        print(f"📶 Final Connectivity Ratio: {final_episode['connectivity_ratio']:.1%}")
        print(f"⚡ Final System Efficiency: {final_episode['system_efficiency']:.1%}")
        print(f"🔄 Final Handover Count: {final_episode['total_handovers']}")
        
        # Calculate improvements over training
        if len(metrics.episode_metrics) > 100:
            early_episodes = metrics.episode_metrics[:100]
            late_episodes = metrics.episode_metrics[-100:]
            
            early_success = np.mean([e['handover_success_rate'] for e in early_episodes])
            late_success = np.mean([e['handover_success_rate'] for e in late_episodes])
            success_improvement = ((late_success - early_success) / (early_success + 1e-6)) * 100
            
            early_efficiency = np.mean([e['system_efficiency'] for e in early_episodes])
            late_efficiency = np.mean([e['system_efficiency'] for e in late_episodes])
            efficiency_improvement = ((late_efficiency - early_efficiency) / (early_efficiency + 1e-6)) * 100
            
            print("="*60)
            print("📈 LEARNING IMPROVEMENTS (First 100 vs Last 100 Episodes)")
            print(f"🎯 Success Rate Improvement: {success_improvement:+.1f}%")
            print(f"⚡ Efficiency Improvement: {efficiency_improvement:+.1f}%")
    
    print("="*60)
    print("📁 Generated Files:")
    print("  📊 training_progress.png - Training curves and progress")
    print("  📈 detailed_performance_metrics.png - Detailed system metrics")
    print("  🏆 baseline_comparison.png - RL vs baseline strategies")
    print("  📚 learning_progress.png - Learning progress over episodes")
    print("  🧠 trained_agent.pth - Trained DQN model")
    print("="*60)
    print("🎉 Run 'python test_agent.py' to see the trained agent in action!")

if __name__ == "__main__":
    main() 