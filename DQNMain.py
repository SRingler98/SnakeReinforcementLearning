from DQNClasses import DQNLearning
from SnakeEnv import SnakeEnv

# start of main
print("Creating model")
env = SnakeEnv(grid_size=10)

temp_learn = DQNLearning(env=env,
                         target_name=str(str("Snake")),
                         episode_count=500,
                         min_batch_size=1000,
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
