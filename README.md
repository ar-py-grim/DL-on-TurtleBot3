# DL-on-TurtleBot3
DQN and DDQN algo on turtlebot3


**To run the project**

### During Training
1. First launch the Gazebo map
```sh
ros2 launch turtlebot3_gazebo turtlebot3_dqn_{$stage_num}.launch.py
```

2. Run Gazebo environment node
```sh
ros2 run turtlebot3_dqn dqn_gazebo {$stage_num}
```

3. Run DQN environment node
```sh
ros2 run turtlebot3_dqn dqn_environment
```

4. Run DQN agent node
```sh
ros2 run turtlebot3_dqn dqn_agent {$stage_num} {$max_training_episodes}
```

5. Run machine learning graph
```sh
ros2 launch turtlebot3_dqn turtlebot3_results.launch.py 
```

### During Testing
1. First launch the Gazebo map
```sh
ros2 launch turtlebot3_gazebo turtlebot3_dqn_{$stage_num}.launch.py
```

2. Run Gazebo environment node
```sh
ros2 run turtlebot3_dqn dqn_gazebo_test {$stage_num}
```

3. Run DQN environment node
```sh
ros2 run turtlebot3_dqn dqn_environment
```
4. Test traind model 
```sh
ros2 run turtlebot3_dqn ddqn_test {$stage_num}
```

## References
Git repo of turtlebot3 Machine Learnning link https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning/tree/humble
<br />
Turtlebot3 documentation link https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/
