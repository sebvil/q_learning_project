# q_learning_project

Team member: Leandra Nealer and Sebastian Villegas Mejia

## Implementation planning

### Q-learning algorithm

1. Executing the Q-learning algorithm

   For every iteration of the algorithm, given the current state, we use the
   action matrix to select of the possible actions at random. Using the 
   callback function of a subscriber to the `/q_learning/reward` topic, we 
   calculate the Q-value for this action, and put it into the corresponding 
   location in the Q-matrix.

   For testing, we'll call the function that will be used as the callback for the subscriber to the QLearningReward topic directly and check the value of the matrix.

2. Determining when the Q-matrix has converged

    At each iteration of the algorithm, we check if the new Q-value is within 
    some small epsilon of the previous value. If this is true for 20 steps in a
    row, the matrix has converged.

    Testing: Run several times on the same values. If the output matrix is approximately the same (within epsilon), the component is working.

3. Once the Q-matrix has converged, how to determine which actions the robot 
   should take to maximize expected reward

    We will look up in the Q-matrix the action that produces the best reward 
    given the current state, and perform the designated action.

    Testing: we will set the matrix and the current state manually, and check
    that the returned action is the expected one.

### Robot perception

1. Determining the identities and locations of the three colored dumbbells

    Define RGB ranges for red, green, and blue. Use the camera to find when we’re looking at these values. (Basically like the line follower project). Use scan data to obtain x and y values of the position relative to the origin.

    Testing: use the camera to see where the robot thinks each color is.

2. Determining the identities and locations of the three numbered blocks.

    Either figure out how to do some basic number recognition with Python 
    (using OpenCV perhaps) or use concepts that we learn in the lecture
    on Wednesday 2/17.

    Testing: same as above, use the camera to see where the robot thinks each block is.

### Robot manipulation & movement

1. Picking up and putting down the dumbbells with the OpenMANIPULATOR arm

    Use the `moveit`_commander package to control the arm. Move directly in front of the dumbbell. Move arm to some position we’ll determine through experimentation. Close grabber and return to original position. We'll use data from LaserScan to determine the distance to the dumbbell and experiment to determine the exact
    values at which to place the arm.

    Testing: put a dumbbell in front of the robot. Have a function that handles “pick up” and “put down”

2. Navigating to the appropriate locations to pick up and put down the dumbbells

    Have the robot remember its current position. Once we have the x and y values of the goal,  move the robot to the correct position using the differences between the current position and the goal.

    Testing: Have a function that takes in current position and goal position. 
    Run this function for testing


