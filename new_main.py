"""Forward and inverse kinematics checks for the Kinova Gen3 arm."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from forward_kinematics import Check_params


END_EFFECTOR = "EndEffector_Link"
JSON_FILE = Path(__file__).with_name("dh_parameters_7dof.json")
URDF_FILE = Path(__file__).with_name("GEN3_URDF_V12_pybullet.urdf")
POINT_1 = np.array([-0.4498738647, 0.1223887801, 1.0277091265])
POINT_2 = np.array([0.5160262585, 0.2863413095, 0.8865555525])
RUN_GUI = "--direct" not in sys.argv


def dh_transform(a, alpha, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),
         np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha),
         -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0.0, np.sin(alpha), np.cos(alpha), d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def read_dh_data():
    with JSON_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if data.get("num_links") != 7 or len(data.get("dh_parameters", [])) != 7:
        raise ValueError("The Gen3 JSON file must contain seven D-H rows.")
    return data

def get_urdf_pose(joint_angles):
    data = read_dh_data()
    dh_pose = np.eye(4)
    for row, theta in zip(data["dh_parameters"], joint_angles):
        dh_pose = dh_pose @ dh_transform(
            row["a"], row["alpha"], row["d"], theta
        )
    base_to_dh_base = np.array(data["base_to_dh_base_transform"])
    dh_end_to_urdf_end = np.array(
        data["dh_end_to_urdf_end_effector_transform"]
    )
    return dh_pose, base_to_dh_base @ dh_pose @ dh_end_to_urdf_end

# Forward Kinematics and Inverse Kinematics using PYBullet


def get_link_index(robot_id, link_name):
    for joint_idx in range(p.getNumJoints(robot_id)):
        name = p.getJointInfo(robot_id, joint_idx)[12].decode("utf-8")
        if name == link_name:
            return joint_idx
    raise ValueError(f"Link {link_name!r} was not found")


def get_joint_indices(robot_id, num_joints):
    return[
        i for i in range(num_joints)
        if p.getJointInfo(robot_id, i)[2] == p.JOINT_REVOLUTE
    ]


def rotation_error(rotation, quaternion):
    pybullet_rotation = np.array(
        p.getMatrixFromQuaternion(quaternion)
    ).reshape(3, 3)

    relative_rotation = rotation.T @ pybullet_rotation
    cosine = (np.trace(relative_rotation) - 1.0) / 2.0
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def run_forward(robot_id, num_joints):
    joint_angles = np.deg2rad(read_dh_data()["joint_angle_offsets"])

    print("THE NUMBER OF JOINTS", num_joints)

    joint_indices = get_joint_indices(robot_id, num_joints)
    for joint, angle in zip(joint_indices, joint_angles):
        p.resetJointState(robot_id, joint, angle)

    p.stepSimulation()

    for joint, angle in zip(joint_indices, joint_angles):
        p.resetJointState(robot_id, joint, angle)

    end_effector_idx = get_link_index(robot_id, END_EFFECTOR)
    link_state = p.getLinkState(
        robot_id, end_effector_idx, computeForwardKinematics=True
    )

    pos = np.array(link_state[4])
    orientation = link_state[5]
    print("\nEnd-effector position (X, Y, Z):", pos)
    print("End-effector orientation:", orientation)

    dh_pose, urdf_pose = get_urdf_pose(joint_angles)
    position_error = np.linalg.norm(pos - urdf_pose[:3, 3])
    orientation_error = rotation_error(
        urdf_pose[:3, :3], orientation
        )


def run_inverse(robot_id, num_joints):

    end_effector_idx = get_link_index(robot_id, END_EFFECTOR)
    joint_idx = get_joint_indices(robot_id, num_joints)
    point_1 = POINT_1.copy()
    point_2 = POINT_2.copy()
    joint_angles = np.deg2rad(read_dh_data()["joint_angle_offsets"])

    for joint, angle in zip(joint_idx, joint_angles):
        p.resetJointState(robot_id, joint, angle)
    p.stepSimulation()

    box_dim = [0.04, 0.04, 0.04]
    start_marker_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=box_dim,
            rgbaColor=[0.1, 0.9, 0.1, 1.0],
        )
    start_marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=start_marker_shape,
            basePosition=point_1,
        )
    end_marker_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=box_dim,
            rgbaColor=[0.95, 0.1, 0.1, 1.0],
        )
    end_marker_id = p.createMultiBody(
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
    p.setRealTimeSimulation(0)

    for frame in frames:
        ik_angles = p.calculateInverseKinematics(
            robot_id,
            end_effector_idx,
            targetPosition = frame.tolist(),
            maxNumIterations=1000,
            residualThreshold =1e-10,
        )
        for joint, angle in zip(joint_idx, ik_angles):
            p.setJointMotorControl2(
                robot_id,
                joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                force=500,
                positionGain=0.5,
            )
        for i in range(30):
            p.stepSimulation()
            if RUN_GUI:
                time.sleep(0.01)

    for i in range(120):
        p.stepSimulation()
        if RUN_GUI:
            time.sleep(0.01)
    
    link_state = p.getLinkState(
            robot_id, end_effector_idx, computeForwardKinematics=True
        )
    end_position = np.array(link_state[4])
    error = np.linalg.norm(point_2 - end_position)
    print("\nTarget end point was: ", point_2)
    print("Achieved end position: ", end_position)
    print("Final position error: ", f"{error:.6e}")


# The run part of the program (connecting and loading the robot )
# using the template code

def main():
    client = p.connect(p.GUI if RUN_GUI else p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.loadURDF("plane.urdf")

        if not URDF_FILE.is_file():
            raise FileNotFoundError(
                f"Prepared robot file was not found: {URDF_FILE.name}"
            )
        robot_id = p.loadURDF(str(URDF_FILE), useFixedBase=True)
        num_joints = p.getNumJoints(robot_id)

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
                run_forward(robot_id, num_joints)
            elif ans == "3":
                run_inverse(robot_id, num_joints)
            elif ans == "4":
                print("The program is terminated")
                loop = False
                break

            else:
                print("Invalid choice, try again.")
    finally:
        p.disconnect(client)


if __name__ == "__main__":
    main()



