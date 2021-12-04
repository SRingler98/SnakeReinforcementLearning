from DQNClasses import DQNLearning
from SnakeEnv import SnakeEngine

# start of main
print("Creating model")
env = SnakeEngine()
temp_learn = DQNLearning(env=env,
                         target_name=str(str(target[0]) + "," + str(target[1])),
                         episode_count=10,
                         min_batch_size=50,
                         max_batch_size=-1,
                         load_model=False,
                         fit_on_step=10,
                         train=True,
                         save_model=False,
                         show_graphs=False)

print("Training model")
temp_agent = temp_learn.train(debug=False,
                              # replay_buffer_data=replay_data[target_index]
                              )
