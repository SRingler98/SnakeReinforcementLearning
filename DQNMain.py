from DQNClasses import DQNLearning
from SnakeEnv import SnakeEnv
from DQNClasses import load_replay_data_from_csv

# start of main

print("Creating model")
env = SnakeEnv(grid_size=10)

temp_learn = DQNLearning(env=env,
                         target_name=str(str("SnakeEpisodic2")),
                         episode_count=10000,
                         min_batch_size=2500,
                         max_batch_size=-1,
                         load_model=False,
                         fit_on_step=2,
                         train=True,
                         save_model=True,
                         show_graphs=True)

print("Training model")
temp_agent = temp_learn.train(debug=False,
                              # replay_buffer_data=replay_data[target_index]
                              )

temp_learn.evaluate(agent=temp_agent,
                    num_of_times=10)
