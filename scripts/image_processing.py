#! /usr/bin/env python3

import cv_bridge
import keras_ocr
import numpy
import rospy
from cv2 import cv2
from sensor_msgs.msg import Image


class ImageProcessing:
    def __init__(self, robot_controller):
        self.robot_controller = robot_controller
        self.image = None

        # locations of the dumbbells and blocks as (x, y) tuples.
        self.db_locs = []
        self.block_locs = []

        # angle of the dumbbells and blocks with respect to the origin
        self.db_thetas = []
        self.block_thetas = []

        # order in which dumbbells and blocks are found
        self.order_db = []
        self.order_blocks = []

        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber(
            "camera/rgb/image_raw", Image, self.image_callback
        )

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def find_block_order(self):
        """Gets the order and locations in which the blocks appear."""
        # use keras-ocr to find block order
        if not self.order_blocks:
            print("Finding blocks... ")
            # initialize the keras pipeline
            pipeline = keras_ocr.pipeline.Pipeline()

            # assembling images of the 3 blocks individually
            self.images = []
            # turn to face each block and get an image of each block
            for angle in self.block_thetas:
                self.robot_controller.turn(angle * 0.01745329)
                rospy.sleep(2)
                self.images.append(self.image)

            # call the recognizer on the list of images
            prediction_groups = pipeline.recognize(self.images)

            for i in range(3):
                word = prediction_groups[i][0][0]
                if word == "l":
                    word = "1"
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
            for i in range(3):
                angle = self.block_thetas[i]
                d = self.block_dist[i]
                if angle > 180:
                    self.db_thetas[i] -= 360
                    angle -= 360

                angle_rad = angle * 0.01745329

                # x and y coordinates of the blocks
                x = d * numpy.cos(angle_rad)
                y = d * numpy.sin(angle_rad)
                self.block_locs.append((x, y))
            print("Block angles: ", self.block_thetas)

    def find_block_thetas(self):
        """Finds the angles from the origin at which the blocks are located."""
        ranges = self.robot_controller.ranges
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
        # image = self.image
        # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # define the upper and lower bounds for what should be considered red,
        # green, and blue
        # red = numpy.uint8([[[0, 0, 255]]])
        # hsvRed = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
        # lower_red = hsvRed[0][0][0] - 10, 100, 100
        # upper_red = hsvRed[0][0][0] + 10, 255, 255
        # lower_red = numpy.array([lower_red[0], lower_red[1], lower_red[2]])
        # upper_red = numpy.array([upper_red[0], upper_red[1], upper_red[2]])

        # green = numpy.uint8([[[0, 255, 0]]])
        # hsvGreen = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
        # lower_green = hsvGreen[0][0][0] - 10, 100, 100
        # upper_green = hsvGreen[0][0][0] + 10, 255, 255
        # lower_green = numpy.array(
        #     [lower_green[0], lower_green[1], lower_green[2]]
        # )
        # upper_green = numpy.array(
        #     [upper_green[0], upper_green[1], upper_green[2]]
        # )

        # blue = numpy.uint8([[[255, 0, 0]]])
        # hsvBlue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
        # lower_blue = hsvBlue[0][0][0] - 10, 100, 100
        # upper_blue = hsvBlue[0][0][0] + 10, 255, 255
        # lower_blue = numpy.array([lower_blue[0], lower_blue[1],
        # [2]])
        # upper_blue = numpy.array([upper_blue[0], upper_blue[1],
        # upper_blue[2]])

        # lower = [lower_red, lower_green, lower_blue]
        # upper = [upper_red, upper_green, upper_blue]
        # get the order of the colors

        # Gets centers of red, green, and blue dumbbells, in that order
        for i in range(3):
            self.db_locs.append(self.get_center_for_color)

        # if one of the colors was not found, assume location is at edge of
        # image
        if -1 in self.db_locs:
            ind = self.db_locs.index(-1)
            avg = (sum(self.db_locs) + 1) / 2
            image_width = self.image.shape[2]
            if avg > image_width / 2:
                self.db_locs[ind] = 0
            else:
                self.db_locs[ind] = image_width

    def get_center_for_color(self, color: int):
        """Gets the center of the image for the given color.

        Parameters:
            color: 0 for red, 1 for green, 2 for blue.
        Returns:
            The x coordinate of the center of the color in the current image.
            If no pixels of the color are found, returns -1.
        """
        image = self.image
        h, w, d1 = image.shape
        search_top = int(h / 2)
        search_bot = int(h / 2 + 1)
        color = numpy.uint8([[[0, 0, 0]]])
        color[0][0][2 - color] = 255
        hsvColor = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        lower_color = numpy.array([hsvColor[0][0][0] - 10, 100, 100])
        upper_color = numpy.array([hsvColor[0][0][0] + 10, 255, 255])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Erase all pixels that aren't the correct color
        mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0

        # Determine the center of the dumbbell
        M = cv2.moments(mask)
        print(M["m00"])
        # Get the center of the dumbbell if color pixels are found
        if M["m00"] > 0:
            # center of the colored pixels in the image
            return int(M["m10"] / M["m00"])

        return -1

    def find_db_locs(self):
        """Find the location of each of the dumbbells."""
        ranges = self.robot_controller.ranges
        # get the initial x and y values of the dumbbells
        if self.order_db and not self.db_locs:
            print("getting x y...")
            # get the order the colors appear in
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

                x = d * numpy.cos(angle_rad)
                y = d * numpy.sin(angle_rad)
                self.db_locs.append((x, y))  # store the three x,y values
            print("DB Angles: ", self.db_thetas, "DB locs:", self.db_locs)

    def analyze_surroundings(self):
        """Determines initial locations of blocks and dumbbells."""
        self.find_db_order()
        self.find_db_locs()

        self.turn(numpy.pi)
        rospy.sleep(1)
        self.find_block_thetas(self.ranges)
        self.find_block_order()

        print("DONE")
