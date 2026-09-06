import json
from pathlib import Path
import numpy as np

# Get the number of paramters and check them

def Check_params():
    
    while True:
        try:
            num_links = int(input("Enter number of links: "))
            if num_links <= 0:  
                raise ValueError
            break
        except ValueError:
            print("Please enter a positive whole number.")

    #Reading the DH paramters from Json file

    file_path = Path(__file__).with_name("dh_parameters_7dof.json")
    print(f"Reading DH parameters from: {file_path.name}")
    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read the JSON file: {error}")
        return

    # Check that the given links is the correct parameter of the manuplator.
    
    if data.get("num_links") != num_links:
        print(f"Error: Entered links:{num_links},and JSON-file contain {data.get('num_links')} links.")
        return
    dh_table = np.zeros((num_links, 4))

    #  Read dh parameters from JSON data.
    
    dh_data = data["dh_parameters"]

    if len(dh_data) != num_links:
        print("Error: the number of D-H parameter rows is incorrect.")
        return

    try:
        for i in range(num_links):
            dh_table[i, 0] = dh_data[i]["link_index"]
            dh_table[i, 1] = dh_data[i]["a"]
            dh_table[i, 2] = dh_data[i]["alpha"]
            dh_table[i, 3] = dh_data[i]["d"]
    except (KeyError, TypeError, ValueError) as error:
        print(f"Error: invalid DH parameter data: {error}")
        return

    
    #Take joint angles as input from the DH table of json file 
    with open("dh_parameters_7dof.json", "r") as file:
        data = json.load(file)
    num_links = data["num_links"]

    # DH parameters
    dh_params = data["dh_parameters"]

    # Actual joint angle
    joint_angles = np.array(data["joint_angle_offsets"], dtype=float)
    for i in range(num_links):
        a = dh_params[i]["a"]
        alpha = dh_params[i]["alpha"]
        d = dh_params[i]["d"]
        theta = joint_angles[i]



    # Print completed DH table.

    np.set_printoptions(precision=4, suppress=True)
    print("\nDH Table")
    print("Columns: [link_index, a, alpha (radians), d]")
    print(dh_table)

    #Build and combine homogeneous transformation matrices using standard-DH
    

    homo_0 = np.eye(4)

    for i in range(num_links):
        a = dh_table[i, 1]
        alpha = dh_table[i, 2]
        d = dh_table[i, 3]
        theta = joint_angles[i]

        homo_i = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha),
             np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha),
             -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1],
        ])

        print(f"\nTransformation matrix for link {i + 1} (A_{i + 1}):")
        print(homo_i)

        homo_0 = homo_0 @ homo_i

    
    # Print final result
    
    print("\nHomogeneous Transformation Matrix (Base to End-Effector):")
    print(homo_0)

    end_pos = homo_0[:3, 3]
    print("End-effector position (Px, Py, Pz):")
    print(end_pos)

    # The Gen3 URDF uses different base and end-effector frame definitions from the D-H frames. To compare this:
    
    if (
        "base_to_dh_base_transform" in data
        and "dh_end_to_urdf_end_effector_transform" in data
    ):
        base_to_dh_base = np.array(data["base_to_dh_base_transform"])
        dh_end_to_urdf_end = np.array(
            data["dh_end_to_urdf_end_effector_transform"]
        )
        urdf_pose = base_to_dh_base @ homo_0 @ dh_end_to_urdf_end
        print("\nConverted pose in the original URDF frame")
        print("(base_link to EndEffector_Link):")
        print(urdf_pose)
        print("Converted end-effector position (Px, Py, Pz):", urdf_pose[:3, 3])


if __name__ == "__main__":
    Check_params()
