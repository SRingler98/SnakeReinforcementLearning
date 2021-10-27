# SnakeRL

Authors: Shawn Ringler and Frankie Torres

<img src="https://www.worldlandtrust.org/wp-content/uploads/2017/10/snake-banner-guatemala.jpg" width="900">

## Intro

This project contains implementations of Reinforcement Learning methods for the classic game Snake. We have implemented our own Snake game 
in Python using Pygame so that we have full control over everything. So far we have only implemented Q-Learning. In the future we plan to implement SARSA, 
DQN, and other DQN variations. We also plan to implement and experiment with different methods for designing the state-space 
as well as the reward function. Our goal is to see and compare the results of implementing these different methods. Ultimately, we hope to 
make a reinforcement learning agent that learns in a feasible amount of time as well as being able to achieve very high average scores that rival
the average scores of a human. The program we are coding will allow the user to play Snake themselves or play with our reinforcement
learning methods to train agents, save and load agents, and graph performance.

## Contents (Python Files)

### main.py
-Simply imports everything needed and runs the program menu

### Menu.py
-A text-based navigation menu that lets the user play Snake themselves or run reinforcment learning algorithms

### SnakeEngine.py
-Our implementation of Snake

### DisplayGrid.py
-Handles the frontend display of the SnakeGame

### QLearning.py
-Has our implementation of Q-Learning for Snake

### Graphing.py
-Methods to graph and measure the performance of QLearning (so far)

## Contents (Folders)

### saved_tables
-Contains saved Q-tables in .json format
