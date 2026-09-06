"""
AR523 - Assignment 1 - Kinova Gen3 7-DOF manipulator.

Part A: D-H forward kinematics
Part B, Section I: PyBullet forward kinematics check
Part B, Section II: PyBullet inverse kinematics and straight-line trajectory
"""

import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from forward_kinematics import Check_params
from prepare_gen3_urdf import prepare_urdf


END_EFFECTOR_LINK_NAME = "EndEffector_Link"
JOINT_ANGLES_DEGREES = np.array(
    [10.0, -20.0, 25.0, -30.0, 15.0, 20.0, -10.0]
)
JOINT_ANGLES_RADIANS = np.deg2rad(JOINT_ANGLES_DEGREES)
POINT_1 = np.array([-0.4498738647, 0.1223887801, 1.0277091265])
POINT_2 = np.array([0.5160262585, 0.2863413095, 0.8865555525])
RUNNING_GUI = "--direct" not in sys.argv
JSON_FILE = Path(__file__).with_name("dh_parameters_7dof.json")


# =========================================================================
# PART A: D-H FORWARD KINEMATICS
# =========================================================================
def run_dh_forward_kinematics():
    """Run the D-H calculation implemented in forward_kinematics.py."""
    Check_params()


def dh_transformation(a, alpha, d, theta):
    """Return a standard-D-H transform: Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),
         np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha),
         -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0.0, np.sin(alpha), np.cos(alpha), d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def load_dh_data():
    """Load and validate the assignment's seven-row D-H parameter file."""
    import json

    with JSON_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if data.get("num_links") != 7 or len(data.get("dh_parameters", [])) != 7:
        raise ValueError("The Gen3 JSON file must contain seven D-H rows.")
    return data


def analytical_urdf_pose(joint_angles_radians):
    """Return the D-H pose and its equivalent URDF-frame pose."""
    data = load_dh_data()
    dh_pose = np.eye(4)
    for row, theta in zip(data["dh_parameters"], joint_angles_radians):
        dh_pose = dh_pose @ dh_transformation(
            row["a"], row["alpha"], row["d"], theta
        )
    base_to_dh_base = np.array(data["base_to_dh_base_transform"])
    dh_end_to_urdf_end = np.array(
        data["dh_end_to_urdf_end_effector_transform"]
    )
    return dh_pose, base_to_dh_base @ dh_pose @ dh_end_to_urdf_end


# =========================================================================
# PART B: FORWARD AND INVERSE KINEMATICS USING PYBULLET
# =========================================================================
def find_link_index(robot_id, link_name):
    """Find a PyBullet link index by its child-link name."""
    for joint_index in range(p.getNumJoints(robot_id)):
        name = p.getJointInfo(robot_id, joint_index)[12].decode("utf-8")
        if name == link_name:
            return joint_index
    raise ValueError(f"Link {link_name!r} was not found.")


def get_actuator_indices(robot_id, num_joints):
    """Return movable revolute joint indices, excluding the fixed tool joint."""
    return [
        i for i in range(num_joints)
        if p.getJointInfo(robot_id, i)[2] == p.JOINT_REVOLUTE
    ]


def apply_robot_visuals(robot_id, num_joints):
    """Give the loaded Gen3 a varied metal, polymer, and actuator finish."""
    link_colors = {
        -1: ([0.06, 0.07, 0.08, 1.0], [0.35, 0.38, 0.42, 1.0]),
        0: ([0.05, 0.30, 0.78, 1.0], [0.18, 0.35, 0.65, 1.0]),
        1: ([0.025, 0.03, 0.035, 1.0], [0.22, 0.24, 0.27, 1.0]),
        2: ([0.82, 0.08, 0.08, 1.0], [0.62, 0.12, 0.12, 1.0]),
        3: ([0.025, 0.03, 0.035, 1.0], [0.22, 0.24, 0.27, 1.0]),
        4: ([0.95, 0.52, 0.05, 1.0], [0.72, 0.40, 0.08, 1.0]),
        5: ([0.025, 0.03, 0.035, 1.0], [0.22, 0.24, 0.27, 1.0]),
        6: ([0.02, 0.56, 0.52, 1.0], [0.12, 0.40, 0.40, 1.0]),
    }

    for link_index in range(-1, num_joints):
        rgba_color, specular_color = link_colors.get(
            link_index, ([0.5, 0.55, 0.58, 1.0], [0.45, 0.47, 0.49, 1.0])
        )
        p.changeVisualShape(
            robot_id,
            linkIndex=link_index,
            rgbaColor=rgba_color,
            specularColor=specular_color,
        )


def rotation_error_degrees(analytical_rotation, pybullet_quaternion):
    """Return the angular difference between two rotation matrices."""
    pybullet_rotation = np.array(
        p.getMatrixFromQuaternion(pybullet_quaternion)
    ).reshape(3, 3)
    relative_rotation = analytical_rotation.T @ pybullet_rotation
    cosine_angle = (np.trace(relative_rotation) - 1.0) / 2.0
    return np.rad2deg(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


# =========================================================================
# PART B, SECTION I: FORWARD KINEMATICS CHECK IN PYBULLET
# =========================================================================
def run_fk_check(robot_id, num_joints):
    """Apply the Part A configuration and compare PyBullet FK with D-H FK."""
    print("\nNumber of joints:", num_joints)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        print(f"Joint {i}: name={info[1].decode('utf-8')}, type={info[2]}")

    actuator_indices = get_actuator_indices(robot_id, num_joints)
    if len(actuator_indices) != len(JOINT_ANGLES_RADIANS):
        raise RuntimeError("Expected seven movable actuator joints in the Gen3 URDF.")
    for joint_index, angle in zip(actuator_indices, JOINT_ANGLES_RADIANS):
        p.resetJointState(robot_id, joint_index, angle)
    p.stepSimulation()
    for joint_index, angle in zip(actuator_indices, JOINT_ANGLES_RADIANS):
        p.resetJointState(robot_id, joint_index, angle)

    end_effector_index = find_link_index(robot_id, END_EFFECTOR_LINK_NAME)
    link_state = p.getLinkState(
        robot_id, end_effector_index, computeForwardKinematics=True
    )
    position = np.array(link_state[4])
    orientation_quat = link_state[5]
    print("\nEnd-effector position (X, Y, Z):", position)
    print("End-effector orientation (quaternion):", orientation_quat)

    dh_pose, analytical_pose = analytical_urdf_pose(JOINT_ANGLES_RADIANS)
    position_error = np.linalg.norm(position - analytical_pose[:3, 3])
    orientation_error = rotation_error_degrees(
        analytical_pose[:3, :3], orientation_quat
    )
    np.set_printoptions(precision=10, suppress=True)
    print("\nJoint angles used (degrees):", JOINT_ANGLES_DEGREES)
    print("\nAnalytical D-H pose (D-H base to D-H frame 7):")
    print(dh_pose)
    print("\nAnalytical pose converted to URDF frames")
    print("(base_link to EndEffector_Link):")
    print(analytical_pose)
    print("\nPosition error norm (m):", f"{position_error:.12e}")
    print("Orientation error (degrees):", f"{orientation_error:.12e}")


# =========================================================================
# PART B, SECTION II: INVERSE KINEMATICS + STRAIGHT-LINE TRAJECTORY
# =========================================================================
def run_ik_straight_line(robot_id, num_joints):
    """Move the end effector between two reachable points using IK."""
    end_effector_index = find_link_index(robot_id, END_EFFECTOR_LINK_NAME)
    actuator_indices = get_actuator_indices(robot_id, num_joints)
    point_1 = POINT_1.copy()
    point_2 = POINT_2.copy()

    for joint_index, angle in zip(actuator_indices, JOINT_ANGLES_RADIANS):
        p.resetJointState(robot_id, joint_index, angle)
    p.stepSimulation()

    box_half_extents = [0.04, 0.04, 0.04]
    start_marker_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=box_half_extents,
        rgbaColor=[0.1, 0.9, 0.1, 1.0],
    )
    start_marker_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=start_marker_shape,
        basePosition=point_1,
    )
    end_marker_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=box_half_extents,
        rgbaColor=[0.95, 0.1, 0.1, 1.0],
    )
    end_marker_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=end_marker_shape,
        basePosition=point_2,
    )
    _ = (start_marker_id, end_marker_id)

    num_waypoints = 30
    waypoints = [
        point_1 + (point_2 - point_1) * (t / (num_waypoints - 1))
        for t in range(num_waypoints)
    ]
    p.setRealTimeSimulation(0)

    for waypoint in waypoints:
        joint_angles = p.calculateInverseKinematics(
            robot_id,
            end_effector_index,
            targetPosition=waypoint.tolist(),
            maxNumIterations=1000,
            residualThreshold=1e-10,
        )
        for joint_index, angle in zip(actuator_indices, joint_angles):
            p.setJointMotorControl2(
                robot_id,
                joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                force=500,
                positionGain=0.5,
            )
        for _ in range(30):
            p.stepSimulation()
            if RUNNING_GUI:
                time.sleep(0.01)

    for _ in range(120):
        p.stepSimulation()
        if RUNNING_GUI:
            time.sleep(0.01)

    link_state = p.getLinkState(
        robot_id, end_effector_index, computeForwardKinematics=True
    )
    achieved_position = np.array(link_state[4])
    error = np.linalg.norm(point_2 - achieved_position)
    print("\nTarget end point was:  ", point_2)
    print("Achieved end position: ", achieved_position)
    print("Final position error (Euclidean norm):", f"{error:.12e}")


# =========================================================================
# MAIN: connect and load the robot once, then show the assignment menu
# =========================================================================
def main():
    physics_client = p.connect(p.GUI if RUNNING_GUI else p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        robot_id = p.loadURDF(str(prepare_urdf()), useFixedBase=True)
        num_joints = p.getNumJoints(robot_id)
        apply_robot_visuals(robot_id, num_joints)

        while True:
            print("\n--- AR523 Assignment 1 ---")
            print("1. Run D-H Forward Kinematics (Part A)")
            print("2. Run PyBullet FK Check (Part B, Section I)")
            print("3. Run PyBullet IK + Straight-Line Trajectory (Part B, Section II)")
            print("4. Exit")
            choice = input("Enter your choice (1-4): ").strip()

            if choice == "1":
                run_dh_forward_kinematics()
            elif choice == "2":
                run_fk_check(robot_id, num_joints)
            elif choice == "3":
                run_ik_straight_line(robot_id, num_joints)
            elif choice == "4":
                break
            else:
                print("Invalid choice, try again.")
    finally:
        p.disconnect(physics_client)


if __name__ == "__main__":
    main()