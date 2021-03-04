# q_learning_project

Team members: Leandra Nealer and Sebastian Villegas Mejia

## Objectives

The purpose of this project is to execute the q-learning algorithm and use its 
output to determine the optimal arrangement of dumbbells in front of numbered 
blocks. After running the algorithm using the `phantom_movement.py` script to
test different orientations, the robot will identify the locations of the 
dumbbells and blocks and then move the dumbbells to the position that maximizes
reward.

## High-Level Description

We first attempt at random the various orientations of blocks and dumbbells
by choosing randomly from the valid actions for the current state. For each 
orientation, the reward is retrieved. Then the q-algorithm equation is used to 
calculate the value of this particular action on this particular state, 
considering both the immediate reward and the possible reward from the next 
state. This process is continued iteratively until convergence (when every 
state has been considered). At this point we have calculated the value of every 
valid action for every state. Using these values we can determine the optimal 
orientation.

## Q-learning algorithm

The code for this section lives in the `q_learning.py` file in the `scripts`
 folder.

### Initializing action matrix

The action matrix is a 2D array indexed by states that holds the action needed to move from state 1 to state 2. (i.e. `action_matrix[state1][state2] = action 
to change from state 1 to state 2`). It holds $-1$ for invalid actions.
Invalid actions are: a state change that requires more than one action, a change 
from a state to itself, any state change that moves a dumbbell from a block to 
origin, any state with more than one dumbbell at a block, and any state that 
moves a dumbbell from a block to another block.

**Random actions:** The possible actions for state `i` are stored at row `i` in
`action_matrix`. At every iteration, the algorithm randomly chooses an action 
from this row that does not have the value $-1$. 

**update_q_matrix():** This is a subscriber function that executes every time a
 reward is published. The algorithm then retrieves the value of the current 
 state and action from the `q_matrix`. It also retrieves the maximum value from 
 the row of the `q_matrix` corresponding to the next state. Then the Q-learning 
 equation is calculated and stored in the cell of the `q_matrix` for the current 
 state and action. The function furthermore checks if the value has changed 
 significantly for convergence purposes.

**Convergence:** The `q_matrix` is determined to have converged when the values 
of the matrix do not change by more than some small epsilon for 50 iterations.
In particular, we use `epsilon = 1` for this project. This condition is checked
in `update_q_matrix` and the counter is either incremented or reset to zero.

**Path execution:** The function `get_opt()` retrieves the optimal action for 
the given state from the `q_matrix`. In other words, it retrieves the index of
the maximum valued cell in the `q_matrix[state]` row. The optimal path is 
determined by executing this action and repeating 2 more times. The `get_opt` 
function is located in `robot_control.py`.
 
### Robot perception

The robot perception code lives in the `image_processing.py` file in the 
`scripts` folder.

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

For identification, we use keras-ocr(source: https://pypi.org/project/keras-ocr/) to identify the numbers on the three blocks. The number on the block is taken and stored in `block_order`. Using the order we have obtained we (similarly to above) edit our `block_thetas` and `block_dist` arrays to reflect the order of the blocks. Now we can index these arrays by block number.

Finally, we do some simple trigonometry to obtain $(x, y)$ values of the three blocks.


### Robot manipulation & movement

The logic for moving the dumbbells is located in the `robot_control.py` script.

Once the locations of the dumbbells and blocks has been determined, moving the
dumbbells to the blocks proceeds as follows:

1. The robot turns to the location of the desired dumbbell, using the `turn` 
   function. The angle is calculated based on the current location of the robot
   and the calculated location of the dumbbell.
2. The robot the approaches the dumbbell, using image data to keep the dumbbell
   centered in the robot's field of view and laser scan data to determine the 
   distance. Steps 1 and 2 are accomplished in the function `move_to_db`.
3. Once the robot is in front of the dumbbell, it simply lifts its arm to carry
   it. 
4. The robot then proceeds to turn to the location of the desired block, again
   using the `turn` function, the current estimated location of the robot, 
   and the calculated location of the block.
5. The robot approached the block. While it is far away, the distance to the 
   block is calculated with the estimated robot pose, but once the robot is 
   close to the block, it proceeds to use laser scan data. This way, it prevents
   distance to other blocks from affecting the estimated distance in the 
   beginning, but uses a more accurate distance based on laser scan in the end.
   Steps 4 and 5 are accomplished in the function `move_to_block`.
6. Once the robot has arrived at the block, it places the dumbbell down, and 
   backs up a bit, to prevent the robot from knocking down the dumbbell when it
   turns.
7. Repeat steps 1-6 for each dumbbell.

The whole sequence of actions, including grabbing and putting down the dumbbell,
is located in the function `process_move`. This function is a subscriber to the
`/q_learning/robot_action` channel. The publishing of moves and loop logic is
done in the function `send_moves`.

## Challenges
* This was my first time intializing 2d-arrays in Python. After a quick google search I used this syntax: `array= [[0]*64] *64`. This syntax apparently creates a shallow list, which causes the entire column to change when an item in one row is changed. This lead to a lot of confusion until I read more about 2d-arrays.
* In writing the turn function, I had to learn to use the robot's odometry to keep track of his rotation. I ended up going back to the particle filter project to see how the `get_yaw_from_euler` function was used, and also reading a lot online about subscribing to the odometry.
* Perception requires a few different ROS topics in tandem to work. I originally wrote the code for this section inside of the subscriber function, and this lead to a lot of messy and unyieldy code. We moved perception to its own script, and set up subscriber functions that simply store the data received so it can be called upon elsewhere. The code is still pretty long, but much better organized
* Getting the matrix to converge was quite difficult because of some concurrency
  issues, but once those issues were solved, it worked well.

## Futurework
* The robot's odometry will become less effective over time as it migrates due to noise. Having some line or marker at the theta=0 position would allow it to correct itself.
* This code assumes the dumbbells are in front and the block are behind it. The bot could instead find the positions of these items all around it first by using scan data to find where items are, and then by using the color and digit recognition to figure out if it is a dumbbell or a block it is seeing.

## Takeaways
* This was the largest program we've written thus far this quarter. We learned a lot about the importance of reorganizing code to readability in a large project, especially when working in pairs or on a team with people who will need to work with your code. 
* The project itself and many of its sub-parts were quite daunting. By breaking these tasks down into many smaller problems with helper functions the task became much more manageable.
* For perception in particular, online resources were essential to this project. When struggling with a new and unfamiliar problem, it's wise to look for resources you can learn from or even implement in your own code to solve your problem.


# Demo

The following recording shows what the robot does after the matrix has converged.
There were a few bugs we did not get to solve, so at the end the robot gets stuck
without going to the next dumbbell, and it takes the green dumbbell to block
3 instead of 1.

![recording.gif](recording.gif)
