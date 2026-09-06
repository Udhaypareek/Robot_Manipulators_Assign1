# AR523 Assignment 1 — Kinova Gen3 Kinematics

## 1. Objective

This assignment implements forward and inverse kinematics for a 7-DOF Kinova
Gen3 manipulator.

- Part A calculates forward kinematics analytically using the standard
  Denavit–Hartenberg (D-H) method.
- Part B validates analytical FK using PyBullet and the supplied Gen3 URDF.
- Part B also uses PyBullet inverse kinematics to move the end effector along
  a straight-line Cartesian trajectory to a visible target point.

## 2. Software requirements

Install the required Python packages once:

```powershell
pip install -r requirements.txt
```

The project uses `numpy` for matrix calculations and `pybullet` for
simulation, forward kinematics, inverse kinematics, and visualization.

## 3. Part A — Analytical D-H forward kinematics

Run:

```powershell
python forward_kinematics.py
```

First enter `7` for the number of links. Then enter the seven joint angles in
degrees. Use this verified reachable configuration for both Part A and Part B:

```text
theta_1 =  10 degrees
theta_2 = -20 degrees
theta_3 =  25 degrees
theta_4 = -30 degrees
theta_5 =  15 degrees
theta_6 =  20 degrees
theta_7 = -10 degrees
```

### How `forward_kinematics.py` works

1. It automatically opens `dh_parameters_7dof.json`; the user is not asked
   for a file path.
2. It reads the number of links and each D-H row: link index, `a`, `alpha`,
   and `d`.
3. It reads the seven variable joint angles `theta_i` from the keyboard and
   converts degrees to radians using `np.deg2rad()`.
4. For every link it builds the standard D-H matrix:

   ```text
   A_i = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i)
   ```

5. It multiplies all matrices in order:

   ```text
   T_0_7 = A_1 A_2 A_3 A_4 A_5 A_6 A_7
   ```

6. It prints the complete D-H table, each `A_i`, the final homogeneous
   transformation matrix, and the end-effector position `(Px, Py, Pz)`.

### D-H table used

The table follows the alpha values supplied in the assignment. The numerical
offsets were derived from the supplied Gen3 URDF using a standard D-H frame
assignment.

| Link | a (m) | alpha (rad) | d (m) |
|---:|---:|---:|---:|
| 1 | 0 | -pi/2 | -0.128380020 |
| 2 | 0 | +pi/2 | -0.011750773 |
| 3 | 0 | -pi/2 | -0.420760023 |
| 4 | 0 | +pi/2 | -0.012750766 |
| 5 | 0 | -pi/2 | -0.314360001 |
| 6 | 0 | +pi/2 | -0.000350489 |
| 7 | 0 | 0 | -0.167455000 |

Negative `d` values indicate the selected D-H z-axis direction; they do not
mean that a physical link has a negative length.

## 4. Why D-H and URDF poses need a frame conversion

The original Gen3 URDF uses its own `base_link` and `EndEffector_Link` frames.
Those are not identical to the D-H base and D-H frame 7 selected for the
analytical derivation. Therefore the JSON file stores two fixed matrices:

- `base_to_dh_base_transform`
- `dh_end_to_urdf_end_effector_transform`

For direct comparison with PyBullet, the code calculates:

```text
T_baseLink_to_endEffector =
    T_baseLink_to_DHBase
    T_DHBase_to_DHFrame7
    T_DHFrame7_to_endEffector
```

This is a coordinate-frame conversion, not an additional robot joint or an
additional D-H row.

## 5. Part B — PyBullet FK and IK

Run:

```powershell
python pybullet_fk_comparison.py
```

The program opens the PyBullet GUI, loads the robot once, and displays a menu:

```text
1. Run D-H Forward Kinematics (Part A)
2. Run PyBullet FK Check (Part B, Section I)
3. Run PyBullet IK + Straight-Line Trajectory (Part B, Section II)
4. Exit
```

### Option 2: PyBullet forward kinematics check

`run_fk_check(robot_id, num_joints)` follows the provided boilerplate:

1. Prints the joint index, name, and type for all URDF joints.
2. Applies the same seven Part A joint angles using `p.resetJointState()`.
3. Finds the end-effector link by its name, `EndEffector_Link`.
4. Calls `p.getLinkState(..., computeForwardKinematics=True)`.
5. Reads position from `link_state[4]` and quaternion orientation from
   `link_state[5]`.
6. Converts the analytical D-H result into URDF frames and prints the
   position/orientation errors.

Expected FK result for the selected angles:

```text
Position error norm: approximately 5.81e-7 m
Orientation error:  approximately 0 degrees
```

The very small position difference is numerical precision/rounding in the
URDF and PyBullet calculations.

### Option 3: PyBullet inverse kinematics

`run_ik_straight_line(robot_id, num_joints)` follows the provided boilerplate:

1. Selects verified reachable start and target points:

   ```text
   Start:  [-0.4498738647, 0.1223887801, 1.0277091265]
   Target: [ 0.5160262585, 0.2863413095, 0.8865555525]
   ```

2. Adds a green visual box at the start point and a red visual box at the
   target point. They are visual-only and do not collide with the robot.
3. Creates 30 linearly interpolated Cartesian waypoints between them.
4. Calls `p.calculateInverseKinematics()` for each waypoint.
5. Commands the seven movable joints using `p.setJointMotorControl2()` in
   `POSITION_CONTROL` mode and repeatedly calls `p.stepSimulation()`.
6. Reads the final end-effector position and reports the Euclidean position
   error relative to the target.

Expected final IK error:

```text
approximately 2.07e-6 m
```

This confirms that the solved joint configuration reaches the selected target.

## 6. File-by-file explanation

| File/folder | Purpose |
|---|---|
| `forward_kinematics.py` | Part A analytical D-H FK program. |
| `dh_parameters_7dof.json` | Gen3 D-H table, units, and fixed frame-conversion matrices. |
| `pybullet_fk_comparison.py` | Main Part B menu program containing both FK validation and IK trajectory code. |
| `prepare_gen3_urdf.py` | Converts ROS `package://` mesh paths in the supplied URDF into local STL paths PyBullet can load. It does not change robot joints or kinematics. |
| `GEN3_URDF_V12.urdf` | Original supplied Kinova Gen3 robot description. |
| `kinova_gen3_meshes/` | Required STL visual meshes for displaying the actual Gen3 arm in PyBullet. |
| `requirements.txt` | Required Python packages. |

## 7. What to submit

Create one ZIP file containing these items:

```text
forward_kinematics.py
dh_parameters_7dof.json
pybullet_fk_comparison.py
prepare_gen3_urdf.py
GEN3_URDF_V12.urdf
kinova_gen3_meshes/                 (include all STL files inside it)
requirements.txt
ASSIGNMENT_DOCUMENTATION.md
```

Optional additions, if requested by the instructor:

- Screenshots of the PyBullet FK pose and IK trajectory.
- Console-output screenshots showing the FK and IK errors.
- A PDF report based on this documentation.

Do **not** submit these generated or obsolete items:

```text
GEN3_URDF_V12_pybullet.urdf          generated automatically at runtime
gen3_dh_matched.urdf                 earlier placeholder model
__pycache__/                         Python cache folder
check.py                             temporary test file, if present
```
