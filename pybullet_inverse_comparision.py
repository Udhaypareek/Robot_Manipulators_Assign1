#FORWARD AND INVERSE KINEMATICS USING PYBULLET

import numpy as np
import pybullet as bull
import pybullet_data
import json
import sys
import time
from pathlib import Path

from forward_kinematics import Check_params
from prepare_gen3_urdf import prepare_urdf


JSON_FILE = Path(__file__).with_name("dh_parameters_7dof.json")
END_EFFECTOR_LINK_NAME = "EndEffector_Link"

# Same manual reachable configuration
try:
    with JSON_FILE.open("r", encoding="utf-8") as file:
        joint_angle_arr = json.load(file)
except (OSError, json.JSONDecodeError) as error:
    print(f"Could not read the JSON file: {error}")
    

JOINT_ANGLES_DEGREES = np.array(joint_angle_arr["joint_angle_offsets"], dtype=float)
JOINT_ANGLES_RADIANS = np.deg2rad(JOINT_ANGLES_DEGREES)

# Start and target points come from two verified reachable Gen3 poses.
POINT_1 = np.array([-0.4498738647, 0.1223887801, 1.0277091265])
POINT_2 = np.array([0.5160262585, 0.2863413095, 0.8865555525])
RUNNING_GUI = "--direct" not in sys.argv


def load_dh_data():
    """Load the Gen3 D-H table and URDF/D-H frame conversions."""
    with JSON_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if data.get("num_links") != 7 or len(data.get("dh_parameters", [])) != 7:
        raise ValueError("The Gen3 JSON file must contain seven D-H rows.")
    return data


def dh_transformation(a, alpha, d, theta):
    """Standard D-H transform: Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),
         np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha),
         -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0.0, np.sin(alpha), np.cos(alpha), d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def analytical_urdf_pose(joint_angles_radians):
    """Return analytical D-H pose and the same pose in URDF frames."""
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


def find_link_index(robot_id, link_name):
    """Find a PyBullet link index using its URDF child-link name."""
    for joint_index in range(bull.getNumJoints(robot_id)):
        name = bull.getJointInfo(robot_id, joint_index)[12].decode("utf-8")
        if name == link_name:
            return joint_index
    raise ValueError(f"Link {link_name!r} was not found.")


def get_actuator_indices(robot_id, num_joints):
    """Get the seven movable joints and exclude the fixed end-effector joint."""
    return [
        i for i in range(num_joints)
        if bull.getJointInfo(robot_id, i)[2] == bull.JOINT_REVOLUTE
    ]


def rotation_error_degrees(analytical_rotation, pybullet_quaternion):
    pybullet_rotation = np.array(
        bull.getMatrixFromQuaternion(pybullet_quaternion)
    ).reshape(3, 3)
    relative_rotation = analytical_rotation.T @ pybullet_rotation
    cosine_angle = (np.trace(relative_rotation) - 1.0) / 2.0
    return np.rad2deg(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


# =========================================================================
# PART B, SECTION I: FORWARD KINEMATICS CHECK IN PYBULLET
# =========================================================================
def run_fk_check(robot_id, num_joints):
    """Apply Part A angles and compare PyBullet FK against analytical FK."""
    # STEP 1: Inspect the joints.
    print("\nNumber of joints:", num_joints)
    for i in range(num_joints):
        info = bull.getJointInfo(robot_id, i)
        print(f"Joint {i}: name={info[1].decode('utf-8')}, type={info[2]}")

    # STEP 2: Set the same joint angles used in Part A.
    actuator_indices = get_actuator_indices(robot_id, num_joints)
    if len(actuator_indices) != len(JOINT_ANGLES_RADIANS):
        raise RuntimeError("Expected seven movable actuator joints in the Gen3 URDF.")
    for joint_index, angle in zip(actuator_indices, JOINT_ANGLES_RADIANS):
        bull.resetJointState(robot_id, joint_index, angle)
    bull.stepSimulation()
    # The required simulation step includes gravity. Restore the exact
    # selected configuration before reading FK, so it matches Part A.
    for joint_index, angle in zip(actuator_indices, JOINT_ANGLES_RADIANS):
        bull.resetJointState(robot_id, joint_index, angle)

    # STEP 3: Identify the end-effector link.
    end_effector_index = find_link_index(robot_id, END_EFFECTOR_LINK_NAME)

    # STEP 4: Read the forward-kinematics result from PyBullet.
    link_state = bull.getLinkState(
        robot_id, end_effector_index, computeForwardKinematics=True
    )
    position = np.array(link_state[4])
    orientation_quat = link_state[5]
    print("\nEnd-effector position (X, Y, Z):", position)
    print("End-effector orientation (quaternion):", orientation_quat)

    # STEP 5: Compare against Part A.
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



# INVERSE KINEMATICS + STRAIGHT-LINE TRAJECTORY

def run_ik_straight_line(robot_id, num_joints):
    """Use IK to move between two reachable points through a straight path."""

    end_effector_index = find_link_index(robot_id, END_EFFECTOR_LINK_NAME)
    actuator_indices = get_actuator_indices(robot_id, num_joints)

    # Define two reachable workspace points.
    point_1 = POINT_1.copy()
    point_2 = POINT_2.copy()
    for joint_index, angle in zip(actuator_indices, JOINT_ANGLES_RADIANS):
        bull.resetJointState(robot_id, joint_index, angle)
    bull.stepSimulation()

    #Place purely visual, non-colliding start/end markers.
    box_half_extents = [0.03, 0.03, 0.03]
    start_marker_shape = bull.createVisualShape(
        shapeType=bull.GEOM_BOX,
        halfExtents=box_half_extents,
        rgbaColor=[0.1, 0.9, 0.1, 1.0],
    )
    start_marker_id = bull.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=start_marker_shape,
        basePosition=point_1,
    )
    end_marker_shape = bull.createVisualShape(
        shapeType=bull.GEOM_BOX,
        halfExtents=box_half_extents,
        rgbaColor=[0.95, 0.1, 0.1, 1.0],
    )
    end_marker_id = bull.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=end_marker_shape,
        basePosition=point_2,
    )
    _ = (start_marker_id, end_marker_id)


    #Generate linear-interpolation waypoints.
    num_waypoints = 30
    waypoints = [
        point_1 + (point_2 - point_1) * (t / (num_waypoints - 1))
        for t in range(num_waypoints)
    ]
    bull.setRealTimeSimulation(0)

    # Solve IK and command motors at every waypoint.
    for wp in waypoints:
        joint_angles = bull.calculateInverseKinematics(
            robot_id,
            end_effector_index,
            targetPosition=wp.tolist(),
            maxNumIterations=1000,
            residualThreshold=1e-10,
        )
        for joint_index, angle in zip(actuator_indices, joint_angles):
            bull.setJointMotorControl2(
                robot_id,
                joint_index,
                controlMode=bull.POSITION_CONTROL,
                targetPosition=angle,
                force=500,
                positionGain=0.5,
            )
        for _ in range(30):
            bull.stepSimulation()
            if RUNNING_GUI:
                time.sleep(0.01)


    # Allow the motor controllers to settle at the final IK configuration.
    for _ in range(120):
        bull.stepSimulation()
        if RUNNING_GUI:
            time.sleep(0.01)


    # STEP 5: Report final target, achieved point, and error.
    link_state = bull.getLinkState(
        robot_id, end_effector_index, computeForwardKinematics=True
    )
    achieved_position = np.array(link_state[4])
    error = np.linalg.norm(point_2 - achieved_position)
    print("\nTarget end point was:  ", point_2)
    print("Achieved end position: ", achieved_position)
    print("Final position error (Euclidean norm):", f"{error:.12e}")



# MAIN PROGRAM: show a menu


def main():
    physics_client = bull.connect(bull.GUI if RUNNING_GUI else bull.DIRECT)
    try:
        bull.setAdditionalSearchPath(pybullet_data.getDataPath())
        bull.setGravity(0, 0, -9.81)
        bull.loadURDF("plane.urdf")

        # Original Kinova Gen3 URDF; helper resolves its STL mesh paths.

        robot_id = bull.loadURDF(str(prepare_urdf()), useFixedBase=True)
        num_joints = bull.getNumJoints(robot_id)

        while True:
            print("\n--- ROBOT MANUPLATOR KINOVA GEN 3 ---")
            print("1. Run D-H Forward Kinematics (Part A)")
            print("2. Run PyBullet FK Check (Part B, Section I)")
            print("3. Run PyBullet IK + Straight-Line Trajectory (Part B, Section II)")
            print("4. Exit")
            choice = input("Enter your choice (1-4): ").strip()
            if choice == "1":
                Check_params()
            elif choice == "2":
                run_fk_check(robot_id, num_joints)
            elif choice == "3":
                run_ik_straight_line(robot_id, num_joints)
            elif choice == "4":
                break
            else:
                print("Invalid choice, try again.")
    finally:
        bull.disconnect(physics_client)


if __name__ == "__main__":
    main()
