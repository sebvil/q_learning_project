#! /usr/bin/env python3

import math

import moveit_commander
import rospy
from geometry_msgs.msg import Pose, Twist, Vector3
from nav_msgs.msg import Odometry
from q_learning_project.msg import QMatrix, RobotMoveDBToBlock
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

import image_processing

ARM_INIT_POSITION = [0, 0.415, 0.305, -0.73]
ARM_GRAB_POSITION = [0, -0.3, 0.305, -0.73]


def get_yaw_from_pose(p: Pose) -> float:
    """Takes in a Pose object and returns yaw."""

    yaw = euler_from_quaternion(
        [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
    )[2]

    return yaw


class RobotControl:
    def __init__(self):
        rospy.init_node("robot_controller")

        self.pose = None
        self.q_matrix = None
        self.converged = False
        self.ranges = []
        self.processing = image_processing.ImageProcessing(self)

        self.speed_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.odom_subs = rospy.Subscriber("/odom", Odometry, self.process_odom)
        self.q_learning_sub = rospy.Subscriber(
            "/q_learning/q_matrix", QMatrix, self.process_q_matrix
        )

        self.scan_sub = rospy.Subscriber("scan", LaserScan, self.process_scan)

        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")

        self.move_group_gripper = moveit_commander.MoveGroupCommander(
            "gripper"
        )

        self.move_publisher = rospy.Publisher(
            "/q_learning/robot_action", RobotMoveDBToBlock, queue_size=10
        )

        self.move_subscriber = rospy.Subscriber(
            "/q_learning/robot_action", RobotMoveDBToBlock, self.process_move
        )

        self.ready = False

    def process_scan(self, data):
        self.ranges = data.ranges

    def process_odom(self, data):
        self.pose = data.pose.pose

    def process_q_matrix(self, data):
        self.q_matrix = data.q_matrix

    def get_opt(self, state):
        # get the optimal action for a given state
        opt = max(self.q_matrix[state].q_matrix_row)
        return self.q_matrix[state].q_matrix_row.index(opt)

    def send_moves(self):
        state = 0
        for i in range(3):
            while not self.ready:
                pass
            self.ready = False
            action = self.get_opt(state)
            robot_db = action // 3
            block_id = action % 3 + 1

            self.move_publisher.publish(
                RobotMoveDBToBlock(
                    robot_db=self.index_color_map[action // 3],
                    block_id=action % 3 + 1,
                )
            )
            state += (4 ** robot_db) * block_id
        print("Final State:", state)

    def process_move(self, data):
        db_id = data.robot_db
        block_id = data.block_id

        # Move to dumbbell and grab it
        self.move_to_db(db_id)
        rospy.sleep(2)
        self.move_group_arm.go(ARM_GRAB_POSITION, wait=True)
        self.move_group_arm.stop()

        # Move to block and place down dumbbell
        self.move_to_block(block_id)
        rospy.sleep(2)
        self.move_group_arm.go(ARM_INIT_POSITION, wait=True)
        self.move_group_arm.stop()

        # Back up to prevent knocking down dumbbell with turn
        speed = Twist()
        speed.linear.x = -0.1
        self.speed_pub.publish(speed)
        rospy.sleep(2)
        speed.linear.x = 0
        self.speed_pub.publish(speed)
        rospy.sleep(2)
        self.ready = True

    def turn(self, angle):
        # turn robot angle radians using odom
        while not self.pose:
            pass
        angle = angle % math.tau
        yaw = get_yaw_from_pose(self.pose) % math.tau
        k = 0.35
        error = angle - yaw
        while abs(error) > 0.05:
            z = k * error
            if abs(error) > math.pi:
                z *= -1
            self.speed_pub.publish(Vector3(0, 0, 0), Vector3(0, 0, z))
            yaw = get_yaw_from_pose(self.pose) % math.tau
            error = angle - yaw
        self.speed_pub.publish(Vector3(0, 0, 0), Vector3(0, 0, 0))

    def move_to_db(self, db_id):
        """Move to the specified dumbbell."""
        x, y = self.processing.db_locs[self.processing.order_db.index(db_id)]
        current_x = self.pose.position.x
        current_y = self.pose.position.y
        print(x, y, db_id)

        # Turn to face dumbbell
        theta = math.atan((y - current_y) / (x - current_x))
        if y < current_y and x > current_x:
            theta = math.tau + theta
        if x < current_x:
            theta += math.pi
        self.turn(theta)

        distance = self.ranges[0]
        center = 0

        max_speed = 0.1
        # Use scan data to get close to the dumbbell and image data to ensure
        # that the robot approaches to the center of the dumbbell.
        while distance > 0.20 or abs(center) > 3:
            w = self.processing.image.shape[2]
            center = (w / 2) - self.processing.get_center_for_color(db_id)
            distance = min(self.ranges[0:20] + self.ranges[-20:])

            speed = Twist()
            if distance > 0.22 and distance < 3.5 and abs(center) < 100:
                speed.linear.x = min(max_speed, distance * 0.1)

            max_speed = min(1, max_speed + 0.1)
            if center != 0:
                speed.angular.z = center * 0.001

            self.speed_pub.publish(speed)

        print("Moved to ", db_id)

    def move_to_block(self, block_id):
        x, y = self.processing.block_locs[
            self.processing.order_blocks.index(block_id)
        ]
        print(x, y, block_id)

        # turn to face block
        current_x = self.pose.position.x
        current_y = self.pose.position.y
        theta = math.atan((y - current_y) / (x - current_x))

        if y < current_y and x > current_x:
            theta = math.tau + theta
        if x < current_x:
            theta += math.pi

        self.turn(theta)

        rate = rospy.Rate(10)

        max_speed = 0.1
        distance = ((current_x - x) ** 2 + (current_y - y) ** 2) ** 0.5
        while distance > 0.8:
            speed = Twist()

            current_x = self.pose.position.x
            current_y = self.pose.position.y

            # Use scan data to determine distance to block only when robot is
            # close enough that other blocks will not interfere. This will
            # also provide a more accurate distance.
            if distance < 1:
                distance = min(self.ranges[0:10] + self.ranges[-10:])
            else:
                distance = ((current_x - x) ** 2 + (current_y - y) ** 2) ** 0.5

            theta = math.atan((y - current_y) / (x - current_x))

            if y < current_y and x > current_x:
                theta = math.tau + theta
            if x < current_x:
                theta += math.pi

            yaw = get_yaw_from_pose(self.pose) % math.tau
            error = theta - yaw
            z = 0.1 * error
            if abs(error) > math.pi:
                z *= -1

            speed.angular.z = z
            speed.linear.x = min(max_speed, distance * 0.1)
            self.speed_pub.publish(speed)
            max_speed = min(max_speed, max_speed + 0.1)
            rate.sleep()

        # Stop the robot
        self.speed_pub.publish(Twist())

    def run(self):

        # self.db_locs = [
        #     (1, 0.5),
        #     (1, 0),
        #     (1, -0.5),
        # ]  # db locations, locs[i] = (x,y) value of db/block i
        # self.block_locs = [
        #     (-2, 2),
        #     (-2, 0),
        #     (-2, -2),
        # ]  # block locs, ** note: this treats the LEFT side(facing the db's)
        # as the positive x-axis

        # self.order_db = [1, 2, 0]  # the order of db
        # self.order_blocks = [3, 2, 1]

        # retrieving both theta values and (x,y) tuples for blocks and db for
        # more options
        self.processing.analyze_surroundings()
        self.turn(0)

        self.move_group_arm.go(ARM_INIT_POSITION, wait=True)
        self.move_group_arm.stop()

        self.move_group_gripper.go([0.01, 0.01], wait=True)
        self.move_group_gripper.stop()

        self.ready = True

        rospy.spin()


if __name__ == "__main__":
    RobotControl().run()
