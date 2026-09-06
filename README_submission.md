# Kinova Gen3 Forward-Kinematics Assignment

## Submit these files and folder

- `forward_kinematics.py` — Part A analytical standard D-H forward kinematics.
- `dh_parameters_7dof.json` — D-H table derived from the supplied Gen3 URDF.
- `pybullet_fk_comparison.py` — Part B PyBullet vs analytical comparison.
- `prepare_gen3_urdf.py` — resolves URDF mesh paths for PyBullet.
- `GEN3_URDF_V12.urdf` — supplied original Kinova Gen3 URDF.
- `kinova_gen3_meshes/` — the eight required Gen3 STL mesh files.
- `requirements.txt` — Python packages used.

Do not submit `gen3_dh_matched.urdf`; it is an earlier placeholder model.
`GEN3_URDF_V12_pybullet.urdf` is generated automatically at runtime and does
not need to be submitted.

## Run

```powershell
pip install -r requirements.txt
python forward_kinematics.py
python pybullet_fk_comparison.py
```

For both parts, use joint angles (degrees):

```text
10, -20, 25, -30, 15, 20, -10
```

The expected comparison position error is approximately `5.81e-7 m`.

In the PyBullet menu, choose option `2` for the forward-kinematics check or
option `3` for inverse kinematics and the visible green-start/red-target
straight-line trajectory.
