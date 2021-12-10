from DQNClasses import DQNLearning
from SnakeEnv import SnakeEnv
from DQNClasses import load_replay_data_from_csv
# start of main

print("Creating model")
env = SnakeEnv(grid_size=10)

temp_learn = DQNLearning(env=env,
                         target_name=str("SnakeEpisodic12Boolean10Thousand"),
                         episode_count=10000,
                         min_batch_size=500,
                         max_batch_size=-1,
                         epsilon=0.1,
                         load_model=False,
                         fit_on_step=2,
                         train=True,
                         save_model=True,
                         show_graphs=True)

print("Training model")
temp_agent = temp_learn.train(debug=False)

temp_learn.evaluate(agent=temp_agent,
                    num_of_times=10)
