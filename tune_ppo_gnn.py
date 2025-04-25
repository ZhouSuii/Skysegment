\
import optuna
import json
import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
import time
import os

from environment import GraphPartitionEnvironment
from agent_ppo_gnn import GNNPPOAgent
from metrics import evaluate_partition, calculate_partition_weights
from run_experiments import create_test_graph # Or load graph

# --- Configuration ---
NUM_NODES = 10 # Use a moderately sized graph for tuning
NUM_PARTITIONS = 2
N_TRIALS = 200 # Number of Optuna trials to run
TRAINING_EPISODES = 300 # Number of episodes per trial (reduced for faster tuning)
MAX_STEPS_PER_EPISODE = 50 # Max steps per episode
OPTUNA_TIMEOUT = 3600 # Timeout for the entire optimization process (e.g., 1 hour)
RESULTS_DIR = "results"
CONFIGS_DIR = "configs"
BEST_PARAMS_FILE = os.path.join(CONFIGS_DIR, "best_gnn_ppo_params.json")
OPTUNA_RESULTS_FILE = os.path.join(RESULTS_DIR, "optuna_gnn_ppo_results.csv")
OPTUNA_PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(OPTUNA_PLOTS_DIR, exist_ok=True)

# --- Optuna Objective Function ---
def objective(trial: optuna.Trial):
    """Optuna objective function for GNN-PPO hyperparameter tuning."""
    run_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trial {trial.number}: Starting on device {device}")

    # 1. Suggest Hyperparameters
    # Search space designed considering efficiency and impact
    config = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512]), # Wider range
        'ppo_epochs': trial.suggest_int('ppo_epochs', 3, 10), # PPO specific
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]), # Affects GPU util & memory
        'entropy_coef': trial.suggest_float('entropy_coef', 0.0, 0.05), # Regularization
        'value_coef': trial.suggest_float('value_coef', 0.4, 0.6), # Loss weighting
        'gamma': trial.suggest_float('gamma', 0.97, 0.999, log=True), # Discount factor
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.98), # GAE parameter
        'clip_ratio': trial.suggest_float('clip_ratio', 0.1, 0.3), # PPO specific
        'update_frequency': trial.suggest_int('update_frequency', 4, 20), # Steps between updates
        # Fixed for efficiency during tuning
        'jit_compile': True,
        'use_cuda_streams': True,
        'use_tensorboard': False # Disable tensorboard logging for speed
    }
    print(f"Trial {trial.number}: Config = {config}")

    agent = None
    env = None
    try:
        # 2. Setup Environment and Agent
        # Create a different graph for each trial to avoid overfitting to one structure
        graph = create_test_graph(num_nodes=NUM_NODES, seed=trial.number)
        env = GraphPartitionEnvironment(graph, NUM_PARTITIONS, max_steps=MAX_STEPS_PER_EPISODE)
        agent = GNNPPOAgent(graph, NUM_PARTITIONS, config=config)

        # 3. Training Loop
        best_reward_trial = float('-inf')
        final_partition = None

        for e in range(TRAINING_EPISODES):
            episode_start_time = time.time()
            state, _ = env.reset()
            total_reward = 0
            done = False

            for step in range(MAX_STEPS_PER_EPISODE):
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)
                # Ensure reward is a float
                agent.store_transition(float(reward), done)
                state = next_state
                total_reward += reward
                if done:
                    break

            loss = agent.update() # Update agent

            if total_reward > best_reward_trial:
                 best_reward_trial = total_reward

            # Optuna Pruning: Check if the trial should be stopped early based on intermediate reward
            trial.report(total_reward, e)
            if trial.should_prune():
                print(f"Trial {trial.number}: Pruned at episode {e} with reward {total_reward}")
                raise optuna.exceptions.TrialPruned()

            if (e + 1) % 25 == 0: # Print progress periodically
                 print(f"Trial {trial.number} - Episode {e+1}/{TRAINING_EPISODES} - Reward: {total_reward:.2f}, Loss: {loss:.4f}, Time: {time.time()-episode_start_time:.2f}s")


        # 4. Evaluation
        final_partition = env.partition_assignment

        # Handle cases where training might fail or produce invalid partitions
        if final_partition is None or len(np.unique(final_partition)) < NUM_PARTITIONS:
             print(f"Trial {trial.number}: Invalid partition generated. Returning high penalty.")
             return float('inf') # Return a large value to indicate failure (since we minimize)

        eval_results = evaluate_partition(graph, final_partition, NUM_PARTITIONS, print_results=False)
        print(f"Trial {trial.number}: Eval Results = {eval_results}")

        # Define the objective: Minimize normalized edge cut + penalty for imbalance
        # Adjust penalty weight as needed
        objective_value = eval_results["normalized_cut"] + 0.1 * (eval_results["weight_imbalance"] - 1)
        print(f"Trial {trial.number}: Objective Value = {objective_value:.6f}")

        return objective_value

    except Exception as e:
        print(f"Trial {trial.number}: Failed with error: {e}")
        import traceback
        traceback.print_exc()
        # Return a large value to indicate failure
        return float('inf')

    finally:
        # 5. Cleanup GPU memory
        del agent
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Trial {trial.number}: Finished. Duration: {time.time() - run_start_time:.2f}s")


# --- Main Execution ---
if __name__ == "__main__":
    # Create or load study
    study_name = "gnn-ppo-partitioning-optimization"
    # Use a database for persistence if needed, e.g., SQLite
    # storage_name = f"sqlite:///{study_name}.db"
    # study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, ...)
    
    # Simple in-memory study for now
    # Use MedianPruner for early stopping of unpromising trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=TRAINING_EPISODES // 4, interval_steps=10)
    study = optuna.create_study(direction='minimize', pruner=pruner)

    print(f"Starting Optuna optimization with {N_TRIALS} trials (Timeout: {OPTUNA_TIMEOUT}s)...")
    print(f"Graph Size: {NUM_NODES} nodes, {NUM_PARTITIONS} partitions")
    print(f"Training per trial: {TRAINING_EPISODES} episodes, {MAX_STEPS_PER_EPISODE} steps/episode")

    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, gc_after_trial=True) # Enable garbage collection
    except KeyboardInterrupt:
        print("Optimization stopped manually.")
    except Exception as e:
        print(f"An error occurred during optimization: {e}")

    # --- Results ---
    print("\n" + "="*20 + " Optimization Finished! " + "="*20)
    print(f"Number of finished trials: {len(study.trials)}")

    # Find and print the best trial
    try:
        best_trial = study.best_trial
        print("\nBest trial:")
        print(f"  Value (Objective): {best_trial.value:.6f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        # Save best hyperparameters
        with open(BEST_PARAMS_FILE, 'w') as f:
            json.dump(best_trial.params, f, indent=4)
        print(f"\nBest hyperparameters saved to {BEST_PARAMS_FILE}")

    except ValueError:
        print("\nNo trials completed successfully. Could not determine best trial.")
        best_trial = None # Ensure best_trial is defined

    # Save all trial results
    try:
        results_df = study.trials_dataframe()
        results_df.to_csv(OPTUNA_RESULTS_FILE, index=False)
        print(f"Optuna trial results saved to {OPTUNA_RESULTS_FILE}")
    except Exception as e:
        print(f"Could not save trial results dataframe: {e}")


    # Optional: Plot optimization history and parameter importances
    if best_trial: # Only plot if there's a best trial
        try:
            # Check if visualization dependencies are installed
            import plotly
            import kaleido
            
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_importance = optuna.visualization.plot_param_importances(study)
            
            history_path = os.path.join(OPTUNA_PLOTS_DIR, "optuna_gnn_ppo_history.png")
            importance_path = os.path.join(OPTUNA_PLOTS_DIR, "optuna_gnn_ppo_importance.png")
            
            fig_history.write_image(history_path)
            fig_importance.write_image(importance_path)
            print(f"Optuna plots saved to {OPTUNA_PLOTS_DIR}")
            
            # Also show intermediate values if desired
            # fig_intermediate = optuna.visualization.plot_intermediate_values(study)
            # fig_intermediate.write_image(os.path.join(OPTUNA_PLOTS_DIR, "optuna_gnn_ppo_intermediate.png"))

        except ImportError:
            print("\nInstall plotly and kaleido to generate Optuna plots: pip install plotly kaleido")
        except Exception as e:
            print(f"\nCould not generate Optuna plots: {e}")

