#This is the main file 
#Kinova Gen 3 (7DOF)

import numpy as n
import pybullet as bullet
import pybullet_data
import sys 
import time 
from pathlib import Path
import json

from forward_kinematics import Check_params
from prepare_gen3_urdf import prepare_urdf


END_EFFECTOR = "EndEffector_Link"
# reading joint angles from the json file
 
JSON_FILE = Path(__file__).with_name("dh_parameters_7dof.json")
try:
    with JSON_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
except (OSError, json.JSONDecodeError) as error:
    print(f"Could not read the JSON file: {error}")
JOINT_ANGLE_DEGREE = n.array((data["joint_angle_offsets"]), dtype=float)
JOINT_ANGLE_RADIAN = n.deg2rad(JOINT_ANGLE_DEGREE)

#print(JOINT_ANGLE_RADIAN)


POINT_1 = n.array([-0.4498738647, 0.1223887801, 1.0277091265])
POINT_2 = n.array([0.5160262585, 0.2863413095, 0.8865555525])
RUN_GUI = "--direct" not in sys.argv



# the DH transformation matrix 

def dh_transform(a, alpha, d, theta):

    return n.array([
        [n.cos(theta), -n.sin(theta) * n.cos(alpha),
         n.sin(theta) * n.sin(alpha), a * n.cos(theta)],
        [n.sin(theta), n.cos(theta) * n.cos(alpha),
         -n.cos(theta) * n.sin(alpha), a * n.sin(theta)],
        [0.0, n.sin(alpha), n.cos(alpha), d],
        [0.0, 0.0, 0.0, 1.0],
    ])


#loading the DH data from JSON

def dh_data():
    with JSON_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if data.get("num_links") != 7 or len(data.get("dh_parameters", [])) != 7:
        raise ValueError("The Gen3 JSON file must contain seven D-H rows.")
    return data

def urdf_pose(joint_angle_rad):
    ## DH pose  == URDF Pose
    data = dh_data()
    dh_pose = n.eye(4)
    for row, theta in zip(data["dh_parameters"], joint_angle_rad):
        dh_pose = dh_pose @ dh_transform(
            row["a"], row["alpha"], row["d"], theta
        )
    base_to_dh_base = n.array(data["base_to_dh_base_transform"])
    dh_end_to_urdf_end = n.array(
        data["dh_end_to_urdf_end_effector_transform"]
    )
    return dh_pose, base_to_dh_base @ dh_pose @ dh_end_to_urdf_end

# Forward Kinematics and Inverse Kinematics using PYBullet


def Link_indexing(robot_id,link_name):
    for joint_idx in range(bullet.getNumJoints(robot_id)):
        name = bullet.getJointInfo(robot_id,joint_idx)[12].decode("utf-8")
        #print("***name***", name)
        if name == link_name:
            return joint_idx
    raise ValueError(f"Link{link_name} was not found")


def joint_index(robot_id,num_joint):
    return[
        i for i in range(num_joint)
        if bullet.getJointInfo(robot_id,i)[2] == bullet.JOINT_REVOLUTE
    ]

#for calculating the rotational error

def calc_rot_error(rotation,pybullet_space):

    PY_rotation = n.array(
        bullet.getMatrixFromQuaternion(pybullet_space)
    ).reshape(3,3)

    relative_rotation = rotation.T@PY_rotation
    cos = (n.trace(relative_rotation)-1.0)/2.0
    return (n.arccos(n.clip(cos,-1.0,1.0)))

# TO CHECK THE FORWARD KINEMATICS IN THE PROGRAM
def run_forward(robot_id,num_joint):

    print("THE NUMBER OF JOINTS",num_joint)

    for i in range(num_joint):
        info = bullet.getJointInfo(robot_id, i)

    joint_idx = joint_index(robot_id,num_joint)
    for joint,angle in zip(joint_idx,JOINT_ANGLE_RADIAN):
        bullet.resetJointState(robot_id,joint,angle)

    bullet.stepSimulation()

    for joint,angle in zip(joint_idx,JOINT_ANGLE_RADIAN):
        bullet.resetJointState(robot_id,joint,angle)

    end_effector_idx = Link_indexing(robot_id,END_EFFECTOR)
    link_state = bullet.getLinkState(
        robot_id,end_effector_idx,computeForwardKinematics = True
    )

    pos = n.array(link_state[4])
    orientation = link_state[5]
    print("\nEnd-effector position (X, Y, Z):", pos)
    print("End-effector orientation:", orientation)

    dh_pose, urdf_POSE = urdf_pose(JOINT_ANGLE_RADIAN)
    position_error = n.linalg.norm(pos - urdf_POSE[:3, 3])
    orientation_error = calc_rot_error(
        urdf_POSE[:3, :3], orientation
        )

def run_inverse(robot_id,num_joint):

    end_effector_idx = Link_indexing(robot_id,END_EFFECTOR)
    joint_idx = joint_index(robot_id,num_joint)
    point_1 = POINT_1.copy()
    point_2 = POINT_2.copy()

    for joint, angle in zip(joint_idx,JOINT_ANGLE_RADIAN):
        bullet.resetJointState(robot_id,joint,angle)
    bullet.stepSimulation()

    box_dim = [0.04, 0.04, 0.04]
    start_marker_shape = bullet.createVisualShape(
            shapeType=bullet.GEOM_BOX,
            halfExtents=box_dim,
            rgbaColor=[0.1, 0.9, 0.1, 1.0],
        )
    start_marker_id = bullet.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=start_marker_shape,
            basePosition=point_1,
        )
    end_marker_shape = bullet.createVisualShape(
            shapeType=bullet.GEOM_BOX,
            halfExtents=box_dim,
            rgbaColor=[0.95, 0.1, 0.1, 1.0],
        )
    end_marker_id = bullet.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=end_marker_shape,
            basePosition=point_2,
        )
    _ = (start_marker_id,end_marker_id)

    framing_speed = 25
    frames= [
            point_1 + (point_2 - point_1) * (t / ( framing_speed - 1))
            for t in range(framing_speed)
        ]
    bullet.setRealTimeSimulation(0)

    for frame in frames:
        joint_angles = bullet.calculateInverseKinematics(
            robot_id,
            end_effector_idx,
            targetPosition = frame.tolist(),
            maxNumIterations=1000,
            residualThreshold =1e-10,
        )
        for joint, angle in zip(joint_idx, joint_angles):
            bullet.setJointMotorControl2(
                robot_id,
                joint,
                controlMode=bullet.POSITION_CONTROL,
                targetPosition=angle,
                force=500,
                positionGain=0.5,
            )
        for i in range(30):
            bullet.stepSimulation()
            if RUN_GUI:
                time.sleep(0.01)

    for i in range(120):
        bullet.stepSimulation()
        if RUN_GUI:
            time.sleep(0.01)
    
        link_state = bullet.getLinkState(
            robot_id, end_effector_idx, computeForwardKinematics=True
        )
        end_position = n.array(link_state[4])
        error = n.linalg.norm(point_2 - end_position)
        print("\nTarget end point was: ", point_2)
        print("Achieved end position: ", end_position)
        print("Final position error: ", f"{error:.6e}")


# The run part of the program (connecting and loading the robot )
# using the template code

def main():
    client = bullet.connect(bullet.GUI if RUN_GUI else bullet.DIRECT)
    try:
        bullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        bullet.setGravity(0,0,-10)
        bullet.loadURDF("plane.urdf")

        robot_id = bullet.loadURDF(str(prepare_urdf()),useFixedBase =True)
        num_joint = bullet.getNumJoints(robot_id)

        loop = True

        while(loop):
            print("### Choose one of the following option ###")
            print("1. Display DH table and Homogenous transformation calculation")
            print("2. Run Pybullet Forward Kinematics Check")
            print("3.Run Inverse Kinematics animation from source postion to destination")
            print("4.EXIT")

            ans = input("Enter choice: > ")

            if ans == "1":
                Check_params()
            elif ans == "2":
                run_forward(robot_id, num_joint)
            elif ans == "3":
                run_inverse(robot_id, num_joint)
            elif ans == "4":
                print("The program is terminated")
                loop = False
                break

            else:
                print("Invalid choice, try again.")
    finally:
        bullet.disconnect(client)


if __name__ == "__main__":
    main()



