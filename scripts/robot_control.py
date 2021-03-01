#! /usr/bin/env python3

import math

import cv_bridge
import keras_ocr
import numpy
import rospy
from cv2 import cv2
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from q_learning_project.msg import QMatrix
from sensor_msgs.msg import Image, LaserScan
from tf.transformations import euler_from_quaternion


def get_yaw_from_pose(p):
    """ A helper function that takes in a Pose object (geometry_msgs) and returns yaw"""
    yaw = euler_from_quaternion(
        [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
    )[2]

    return yaw


class RobotControl:
    def __init__(self):
        self.pose = None
        self.q_matrix = None
        self.converged = False

        self.image = None
        self.ranges = []
        self.odom = None
        self.db_locs = []  # db locations, locs[i] = (x,y) value of db/block i
        self.block_locs = (
            []
        )  # block locs, ** note: this treats the LEFT side(facing the db's) as the positive x-axis
        self.db_thetas = []  # thetas[i] = theta location of db/block i
        self.block_thetas = []
        self.order_db = []  # the order of db
        self.order_blocks = []

        rospy.init_node("navigators")
        self.speed_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.odom_subs = rospy.Subscriber("/odom", Odometry, self.process_odom)
        self.q_learning_sub = rospy.Subscriber(
            "/q_learning/q_matrix", QMatrix, self.process_q_matrix
        )
        self.bridge = cv_bridge.CvBridge()

        self.image_sub = rospy.Subscriber(
            "camera/rgb/image_raw", Image, self.image_callback
        )

        self.scan_sub = rospy.Subscriber("scan", LaserScan, self.process_scan)

    def get_opt(self, state):
        # get the optimal action for a given state
        opt = max(self.q_matrix[state])
        return self.q_matrix.index(opt)

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def process_scan(self, data):
        self.ranges = data.ranges

    def find_block_order(self):
        # use keras-ocr to find block order
        if not self.order_blocks:
            print("Finding blocks... ")

            # find block order
            # initalize the debugging window
            # cv2.namedWindow("window", 1)

            # initialize the keras pipeline
            pipeline = keras_ocr.pipeline.Pipeline()

            # assembling images of the 3 blocks individually
            self.images = []
            i = 0
            for angle in self.block_thetas:
                self.turn(angle * 0.01745329)
                rospy.sleep(2)
                self.images.append(self.image)
                # cv2.imshow("window", self.images[i])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                i += 1

            # using keras-ocr to get digits
            # source: https://pypi.org/project/keras-ocr/
            # To install from PyPi
            # pip install keras-ocr
            # pip install tensorFlow

            # call the recognizer on the list of images
            prediction_groups = pipeline.recognize(self.images)
            # prediction_groups is a list of predictions for each image
            # prediction_groups[0] is a list of tuples for recognized characters for img1
            # the tuples are of the formate (word, box), where word is the word
            # recognized by the recognizer and box is a rectangle in the image where the recognized words reside

            for i in range(3):
                word = prediction_groups[i][0][0]
                self.order_blocks.append(int(word))

            # reorder theta and dist to reflect order of blocks
            tmp_theta = self.block_thetas.copy()
            tmp_dist = self.block_dist.copy()
            i = 0
            for block in self.order_blocks:
                self.block_thetas[block - 1] = tmp_theta[i]
                self.block_dist[block - 1] = tmp_dist[i]
                i += 1

            # calculate x,y values for 3 blocks
            # note: this treats the LEFT side(facing the db's) as the positive x-axis
            for i in range(3):
                angle = self.block_thetas[i]
                d = self.block_dist[i]
                if angle > 180:
                    self.db_thetas[i] -= 360
                    angle -= 360

                angle_rad = angle * 0.01745329

                x = d * numpy.sin(angle_rad)
                y = d * numpy.cos(angle_rad)
                # print("x: ", x, "y: ", y)
                # print("Angles: ", self.block_thetas)
                self.block_locs.append((x, y))  # store the three x,y values
            print("Block angles: ", self.block_thetas)

    def find_block_thetas(self, ranges):
        # find theta values for the blocks
        block_num = 0
        theta = 90
        self.block_thetas = [-1, -1, -1]
        self.block_dist = [-1, -1, -1]
        while block_num < 3:
            # scan for the 3 blocks
            theta = theta % 360
            if ranges[theta] != numpy.inf:
                self.block_thetas[block_num] = (
                    theta + 180 - 12.7
                )  # subtract to get to ~middle of block
                self.block_dist[block_num] = ranges[theta]
                if self.block_thetas[block_num] >= 180:
                    self.block_thetas[block_num] -= 360
                block_num += 1
                theta -= 25
            theta -= 1

    def find_db_order(self):
        print("Finding dumbbell order... ")
        while self.image is None:
            print("waiting for image...")
            rospy.sleep(1)
        image = self.image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # define the upper and lower bounds for what should be considered red, green, and blue
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
        lower_green = numpy.array(
            [lower_green[0], lower_green[1], lower_green[2]]
        )
        upper_green = numpy.array(
            [upper_green[0], upper_green[1], upper_green[2]]
        )

        blue = numpy.uint8([[[255, 0, 0]]])
        hsvBlue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
        lower_blue = hsvBlue[0][0][0] - 10, 100, 100
        upper_blue = hsvBlue[0][0][0] + 10, 255, 255
        lower_blue = numpy.array([lower_blue[0], lower_blue[1], lower_blue[2]])
        upper_blue = numpy.array([upper_blue[0], upper_blue[1], upper_blue[2]])

        lower = [lower_red, lower_green, lower_blue]
        upper = [upper_red, upper_green, upper_blue]
        if not self.order_db:
            # get the order of the colors
            for i in range(3):
                print(i)
                mask = cv2.inRange(hsv, lower[i], upper[i])

                # this erases all pixels that aren't the correct color
                h, w, d1 = image.shape
                search_top = int(h / 2)
                search_bot = int(h / 2 + 1)

                mask[0:search_top, 0:w] = 0
                mask[search_bot:h, 0:w] = 0

                # # using moments() function, the center of the dumbbell is determined
                M = cv2.moments(mask)
                print(M["m00"])
                # # if there are any color pixels found
                if M["m00"] > 0:
                    # center of the colored pixels in the image
                    cx = int(M["m10"] / M["m00"])
                    # cy = int(M['m01']/M['m00'])

                    # a red circle is visualized in the debugging window to indicate
                    # the center point of the yellow pixels
                    # cv2.circle(image, (cx, cy), 30, (0,0,255), -1)

                    self.order_db.append(cx)

        # cv2.imshow("window", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def find_db_locs(self):
        ranges = self.ranges
        # get the initial x and y values of the dumbbells
        if self.order_db and not self.db_locs:
            print("getting x y...")

            # get the order the colors appear in
            print(self.order_db)
            front = self.order_db.index(numpy.median(self.order_db))
            right = self.order_db.index(numpy.max(self.order_db))
            left = self.order_db.index(numpy.min(self.order_db))

            self.order_db[1] = front
            self.order_db[2] = right
            self.order_db[0] = left

            db = 0
            theta = 90
            # getting the angles of the 3 db's
            self.db_thetas = [-1, -1, -1]
            while db < 3:
                theta = theta % 360
                print(theta)
                if ranges[theta] != numpy.inf:
                    color = self.order_db[db]
                    self.db_thetas[color] = theta
                    db += 1
                    theta -= 25
                theta -= 1

            # calculate x,y values for 3 dbs
            for i in range(3):
                angle = self.db_thetas[i]
                d = ranges[angle]
                if angle > 180:
                    self.db_thetas[i] -= 360
                    angle -= 360

                angle_rad = angle * 0.01745329

                x = d * numpy.sin(angle_rad)
                y = d * numpy.cos(angle_rad)
                print("DB Angles: ", self.db_thetas)
                self.db_locs.append((x, y))  # store the three x,y values
                # the furthest right db has negative xvalue

    def turn(self, angle):
        # turn robot angle radians using odom
        while not self.odom:
            pass
        yaw = get_yaw_from_pose(self.odom)
        k = 0.35
        error = angle - yaw
        while abs(error) > 0.05:
            z = k * error
            self.vel_pub.publish(Vector3(0, 0, 0), Vector3(0, 0, z))
            yaw = get_yaw_from_pose(self.odom)
            error = angle - yaw
        self.vel_pub.publish(Vector3(0, 0, 0), Vector3(0, 0, 0))

    def process_odom(self, data):
        self.pose = data.pose.pose

    def process_q_matrix(self, data):
        if not self.pose or not self.converged:
            return

        self.q_matrix = data.q_matrix

        state = 0
        for i in range(3):
            action = self.get_opt(state)
            robot_db = action // 3
            block_id = action % 3 + 1

            x, y = self.db_locs[self.order_db.index(robot_db)]
            self.move_to_xy(x, y)

            x, y = self.db_locs[self.order_blocks.index(block_id)]
            self.move_to_xy(x, y)

            state += (4 ** robot_db) * block_id
        print("Final State:", state)

    def move_to_xy(self, x, y):
        while True:
            speed = Twist()
            current_x = self.pose.position.x
            current_y = self.pose.position.y
            distance = ((current_x - x) ** 2 + (current_y - y) ** 2) ** 0.5
            if distance < 0.1:
                break
            theta = (
                math.atan((y - current_y) / (x - current_x)) * 180 / math.pi
            )

            yaw = (get_yaw_from_pose(self.pose) * 180 / math.pi) % 360
            if abs(theta - yaw) > 10:
                speed.linear.x = 0.0
                speed.angular.z = abs(theta - yaw) / (theta - yaw) * 0.5
            else:
                speed.linear.x = min(1, distance / 2)
            speed.angular.z = abs(theta - yaw) * 0.05
            self.speed_pub.publish(speed)

    def run(self):

        # retrieving both theta values and (x,y) tuples for blocks and db for
        # more options
        self.find_db_order()
        self.find_db_locs()

        self.turn(numpy.pi)
        rospy.sleep(1)
        self.find_block_thetas(self.ranges)
        self.find_block_order()
        self.turn(0)
        self.converged = True

        self.q_matrix_publisher.publish(QMatrix(q_matrix=self.q_matrix))
        rospy.spin()


if __name__ == "__main__":
    RobotControl().run()
