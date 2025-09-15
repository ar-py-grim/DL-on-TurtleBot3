# DL-on-TurtleBot3
DQN and DDQN algo on turtlebot3

# Software versions
1. Python version = 3.10.12
2. pyqtgraph version = 0.13.7
3. PyQt5 version = 5.15.6
4. Tensorflow, Keras version = 2.10.0
5. numpy version = 1.26.4
   
## Note!
To run the desired turtlebot model source the below line in u .bashrc file
```sh
export TURTLEBOT3_MODEL=burger
```
U can even use other models

## To run the project

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
ros2 run turtlebot3_dqn dqn_test {$stage_num}
```

Follow the same steps for DDQN as well

## References
Git repo of turtlebot3 Machine Learnning link https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning/tree/humble
<br />
Turtlebot3 documentation link https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/
