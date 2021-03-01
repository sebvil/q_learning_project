# q_learning_project

Team members: Leandra Nealer and Sebastian Villegas Mejia

## Objectives
The purpose of this project is to execute the q-learning algoritm and use its output to determine the optimal arrangement of dumbbells. After running the algorithm using the phantom_movement script to test different orientations, the robot will identify the locations of the dumbbells and blocks and then move the dumbbels to the position that maximizes reward.

## High-Level Description
We first attempt at random the various orientations of blocks and dumbbells by choosing randomly from the valid actions for the current state. For each orientation, the reward is retrieved. Then the q-algorithm equation is used to calculate the value of this particular action on this particular state, considering both the immediate reward and the possible reward from the next state. This process is continued iteratively until convergence (when every state has been considered). At this point we have calculated the value of every valid action for every state. Using these values we can determine the optimal orientation.

### Initializing action matrix
The action matrix is a 2D array indexed by states that holds the action needed to move from state 1 to state 2. (i.e. action[state1][state2]) = (action to change from state 1 to state 2). It holds negative 1 for invalid actions.
Invalid actions are: a state change that requires more than one action, a change from a state to itself, any state change that moves a dumbbell from a block to origin, any state with more than one dumbbell at a block, and any state that moves a dumbbell from a block to another block.

### Q-learning algorithm

### Robot perception
The robot perception code lives in the robot_control.py file in the scripts folder.

For perceiving and locating the dumbbells and numbered blocks, we use a combination of the laser scan data and RGB camera data. We have subscriber functions to both of these to get constant updates as the robot turns. The camera is used to determine the order of the items, and then the scan data matches this order to the corresponding physical locations. 

It should be noted that this code takes the robots initial location to be the origin. The robot is initially facing towards the positive y-axis. The positive x-axis is left of the robot. This orientation is convenient use with theta values for rotating.

#### Dumbbells
**find_db_order():** This function draws from the line_follower code to identify the locations on the screen of the 3 colored dumbbells. To determine appropriate color ranges, code from here was used: https://stackoverflow.com/questions/36817133/identifying-the-range-of-a-color-in-hsv-using-opencv. 

The x values retrieved using the line follower code are stored in self.db_order to be reordered and used later
**find_db_locs():** First, we edit the self.db_order array. Now it is indexed by the order of the dumbbell and stores the color (red = 0, green = 1, blue = 2) of the dumbbell in that location.

Then the bot scans 180 degrees in front of it and finds the theta and distance values to the three dumbbells. Some simple trigonometry is performed to obtain (x,y) values. We combine this data with our knowledge about the order of the colors to store in self.db_locs and self.db_thetas the (x,y) values and theta locations, indexed by color.

#### Blocks
The helper function turn() is defined for use in this section. Turn uses the robots odometry to turn to a given theta value on the grid. This part of the code begins with/assumes the robot is turned facing the center center block.

**find_block_thetas():** This function simply scans 180 degrees in front of the robot to obtain the theta values and distances to the three blocks before identifying them.

**find_block_order():** Using the theta values we just found, the bot turns to these three locations and stores the images of the blocks for identification.

For identification we use keras-ocr(source: https://pypi.org/project/keras-ocr/) to identify the numbers on the three blocks. The number on the block is taken and stored in block_order. Using the order we have obtained we (similarly to above) edit our block_thetas and block_dist arrays to reflect the order of the blocks. Now we can index these arrays by block number.

Finally we do some simple trigonometry to obtain (x,y) values of the three blocks.


### Robot manipulation & movement

## Challenges
* This was my first time intializing 2d-arrays in Python. After a quick google search I used this syntax: "array= [[0]*64] *64". This syntax apparently creates a shallow list, which causes the entire column to change when an item in one row is changed. This lead to a lot of confusion until I read more about 2d-arrays.
* In writing the turn function, I had to learn to use the robot's odometry to keep track of his rotation. I ended up going back to the particle filter project to see how the get_yaw_from_euler function was used, and also reading a lot online about subscribing to the odometry.
* Perception requires a few different ROS topics in tandem to work. I originally wrote the code for this section inside of the subscriber function, and this lead to a lot of messy and unyieldy code. We moved perception to its own script, and set up subscriber functions that simply store the data received so it can be called upon elsewhere. The code is still pretty long, but much better organized

## Futurework
* The robot's odometry will become less effective over time as it migrates due to noise. Having some line or marker at the theta=0 position would allow it to correct itself.
* This code assumes the dumbbells are in front and the block are behind it. The bot could instead find the positions of these items all around it first by using scan data to find where items are, and then by using the color and digit recognition to figure out if it is a dumbbell or a block it is seeing.

## Takeaways
* This was the largest program we've written thus far this quarter. We learned a lot about the importance of reorganizing code to readability in a large project, especially when working in pairs or on a team with people who will need to work with your code. 
* The project itself and many of its sub-parts were quite daunting. By breaking these tasks down into many smaller problems with helper functions the task became much more manageable.
* For perception in particular, online resources were essential to this project. When struggling with a new and unfamiliar problem, it's wise to look for resources you can learn from or even implement in your own code to solve your problem.


