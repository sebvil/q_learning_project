#!/usr/bin/env python3

import rospy, cv2, cv_bridge, numpy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Quaternion, Point, Pose, PoseArray, PoseStamped, Twist
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Header, String
from q_learning_project.msg import QLearningReward, RobotMoveDBToBlock, QMatrix
import random
import time
import collections


class QLearning:
    def __init__(self, gamma=0.5, alpha=1, test_mode=False):
        self.counter = 0  # check if q algorithm has converged
        self.gamma = gamma  # discount factor
        self.alpha = alpha  # learning rate
        self.epsilon = 1
        self.db_locs = [] 
        self.theta = []
        self.order = []
        self.converged = True
        self.initialized = False

        # init q matrix
        self.q_matrix = [
            [0] * 9 for j in range(64)
        ]  # q_matrix[i] = the index in the action matrix of optimal state

        self._init_action_matrix()
        if not test_mode:
            rospy.init_node("turtlebot3_q_learning")
            # set up ROS / OpenCV bridge
            self.bridge = cv_bridge.CvBridge()
            self.image_sub = rospy.Subscriber('camera/rgb/image_raw',Image, self.image_callback)

            self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
            self.vel_msg = Twist()

            self.reward_subscriber = rospy.Subscriber(
                "/q_learning/reward", QLearningReward, self.update_q_matrix
            )

            self.scan_sub = rospy.Subscriber('scan', LaserScan, self.process_scan)

            self.move_publisher = rospy.Publisher(
                "/q_learning/robot_action", RobotMoveDBToBlock, queue_size=10
            )
            self.q_matrix_publisher = rospy.Publisher(
                "/q_learning/q_matrix", QMatrix, queue_size=10
            )

            self.q_matrix_publisher.publish(QMatrix(q_matrix=self.q_matrix))

        self.index_color_map = {0: "red", 1: "green", 2: "blue"}
        # ints representing the current state and action taken
        self.action_states_queue = collections.deque()
        self.state = 0
        self.iterations = 0
        self.waiting_for_reward = False

    def _init_action_matrix(self):
        # define red = 0, green = 1, blue = 2
        # define origin = 0, block 1 = 1, ....
        actions = []
        # let actions[i] = (location red, location green, location blue) in state i
        for blue_loc in range(4):
            for green_loc in range(4):
                for red_loc in range(4):
                    actions.append((red_loc, green_loc, blue_loc))

        # creating the action matrix
        self.action_matrix = [[-1] * 64 for j in range(64)]
        for index1, start in enumerate(actions):
            for index2, goal in enumerate(actions):
                # check that goal state is valid
                goal_valid = self._is_goal_valid(goal)
                if not goal_valid:
                    continue
                # check only one dumbell is moving at a time

                if self._is_move_valid(start, goal):
                    for color in range(3):
                        if start[color] < goal[color] and goal[color] != 0:
                            action = goal[color] - 1 + color * 3
                            self.action_matrix[index1][index2] = action

    def _is_goal_valid(self, goal):
        goal_valid = True
        found_pos = set()
        for position in goal:
            if position != 0 and position in found_pos:
                goal_valid = False
                break
            found_pos.add(position)
        return goal_valid

    def _is_move_valid(self, start, goal):
        moves = 0
        for x, y in zip(start, goal):
            if y != x and x == 0:
                moves += 1
            elif y != x:
                return False
        return moves == 1

    def update_q_matrix(self, data):
        """Updates thr Q-matrix based on the give reward."""
        if not self.action_states_queue:
            return
        reward = data.reward
        print("reward: ", reward)
        state, next_state, action = self.action_states_queue.popleft()
        next_actions_diffs = [
            x - self.q_matrix[state][action] for x in self.q_matrix[next_state]
        ]
        new_value = self.q_matrix[state][action] + self.alpha * (
            reward + self.gamma * max(next_actions_diffs)
        )
        if abs(new_value - self.q_matrix[state][action]) < self.epsilon:
            print(
                "new: {}, old: {}, state: {}, counter: {}, reward: {}".format(
                    new_value,
                    self.q_matrix[state][action],
                    state,
                    self.counter,
                    reward,
                )
            )
            self.counter += 1
        else:
            self.counter = 0
        self.q_matrix[state][action] = new_value

        self.q_matrix_publisher.publish(QMatrix(q_matrix=self.q_matrix))
        self.waiting_for_reward = False

    def q_algorithm(self):
        # to do
        self.last_action = -1
        while self.counter < 100:
            #print(self.counter, self.iterations)
            time.sleep(0.5)
            possible_actions = [
                i for i in self.action_matrix[self.state] if i != -1
            ]
            action = random.choice(possible_actions)
            next_state = self.action_matrix[self.state].index(action)
            self.action_states_queue.append((self.state, next_state, action))
            self.iterations += 1

            self.move_publisher.publish(
                RobotMoveDBToBlock(
                    robot_db=self.index_color_map[action // 3],
                    block_id=action % 3 + 1,
                )
            )
            if self.last_action == action:
                print("ACTION REPEATED", self.state, next_state, action)
            self.last_action = action
            if self.iterations % 3 == 0:
                self.state = 0
            else:
                self.state = next_state
        self.converged = True

    def get_opt(self, state):
        # get the optimal action for a given state
        opt = max(self.q_matrix[state])
        return self.q_matrix.index(opt)

    def image_callback(self, msg):
        cv2.namedWindow("window", 1)
        # converts the incoming ROS message to OpenCV format and HSV (hue, saturation, value)
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #define the upper and lower bounds for what should be considered red, green, and blue
        red = numpy.uint8([[[0, 0, 255]]])
        hsvRed = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
        lower_red = hsvRed[0][0][0] - 10, 100, 100
        upper_red = hsvRed[0][0][0] + 10, 255, 255
        lower_red = numpy.array([lower_red[0], lower_red[1], lower_red[2]])
        upper_red = numpy.array([upper_red[0], upper_red[1], upper_red[2]])

        green = numpy.uint8([[[0, 255, 0]]])
        hsvGreen = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
        lower_green = hsvGreen[0][0][0] - 10, 100, 100
        upper_green = hsvGreen[0][0][0] + 10, 255, 255
        lower_green = numpy.array([lower_green[0], lower_green[1], lower_green[2]])
        upper_green = numpy.array([upper_green[0], upper_green[1], upper_green[2]])

        
        blue = numpy.uint8([[[255, 0, 0]]])
        hsvBlue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
        lower_blue = hsvBlue[0][0][0] - 10, 100, 100
        upper_blue = hsvBlue[0][0][0] + 10, 255, 255
        lower_blue = numpy.array([lower_blue[0], lower_blue[1], lower_blue[2]])
        upper_blue = numpy.array([upper_blue[0], upper_blue[1], upper_blue[2]])

        lower = [lower_red, lower_green, lower_blue]
        upper = [upper_red, upper_green, upper_blue]
        if self.converged and not self.order: 
            # get the order of the colors
            for i in range(3):
                mask = cv2.inRange(hsv, lower[i], upper[i])

                # this erases all pixels that aren't the correct color
                h, w, d1 = image.shape
                search_top  = int(h/2)
                search_bot = int(h/2+1)
    
                mask[0:search_top, 0:w] = 0
                mask[search_bot:h, 0:w] = 0

                # # using moments() function, the center of the dumbbell is determined
                M = cv2.moments(mask)

                # # if there are any color pixels found
                if M['m00'] > 0:
                #                 # center of the colored pixels in the image
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])

                #                 # a red circle is visualized in the debugging window to indicate
                #                 # the center point of the yellow pixels
                    cv2.circle(image, (cx, cy), 30, (0,0,255), -1)
    
                    self.order.append(cx)
        cv2.imshow("window", image)
        cv2.waitKey(3)
    
    def process_scan(self, data):
        # get the initial x and y values of the dumbbells
        if self.converged and self.order and not self.db_locs:
            print("getting x y...")
            
            #get the order the colors appear in
            front = self.order.index(numpy.median(self.order))
            right = self.order.index(numpy.max(self.order))
            left = self.order.index(numpy.min(self.order))

            self.order[1] = front
            self.order[2] = right
            self.order[0] = left

            db = 0
            theta = 90
            # getting the angles of the 3 db's
            self.theta = [-1,-1,-1]
            while db < 3:
                if (theta < 0):
                    theta +=360
                if data.ranges[theta] != numpy.inf:
                    color = self.order[db]
                    self.theta[color] = theta
                    db +=1
                    theta-=25
                theta -= 1
           
            # calculate x,y values for 3 dbs
            for i in range(3):
                angle = self.theta[i]
                d = data.ranges[angle]

                if angle > 180:
                    angle -= 360
                angle_rad = angle * 0.01745329
                
                x = d * numpy.sin(angle_rad)
                y = d * numpy.cos(angle_rad)
                print("x: ", x, "y: ", y)
                self.db_locs.append((x,y)) # store the three x,y values
                # the furthest right db has negative xvalue 


    def run(self):
        while not rospy.is_shutdown():
            self.q_algorithm()
            rospy.spin()


if __name__ == "__main__":
    Q = QLearning()
    Q.run()
