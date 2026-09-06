"""Prepare the supplied Kinova Gen3 URDF for PyBullet.

The supplied URDF references mesh files from ROS's ``kortex_description``
package.  Those STL files are not included with the assignment URDF, so
PyBullet cannot load it directly.  This utility removes only visual and
collision tags; it preserves every original link, joint, joint origin, axis,
limit, and inertial property used for forward kinematics.
"""

from pathlib import Path
import xml.etree.ElementTree as ET


WORKSPACE = Path(__file__).resolve().parent
URDF_NAME = "GEN3_URDF_V12.urdf"
OUTPUT_URDF = WORKSPACE / "GEN3_URDF_V12_pybullet.urdf"
MESH_DIRECTORY = WORKSPACE / "kinova_gen3_meshes"
MESH_URI_PREFIX = "package://kortex_description/arms/gen3/7dof/meshes/"


def find_source_urdf():
    """Use a local copy first, then the assignment file in Downloads."""
    possible_paths = [
        WORKSPACE / URDF_NAME,
        Path.home() / "Downloads" / URDF_NAME,
    ]
    for path in possible_paths:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{URDF_NAME} was not found in the assignment folder or Downloads."
    )


def prepare_urdf():
    """Return a PyBullet-loadable URDF with original FK information intact."""
    source_urdf = find_source_urdf()
    root = ET.parse(source_urdf).getroot()

    missing_meshes = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith(MESH_URI_PREFIX):
            local_mesh = MESH_DIRECTORY / filename.removeprefix(MESH_URI_PREFIX)
            if local_mesh.is_file():
                # PyBullet does not understand ROS package:// paths.
                mesh.set("filename", local_mesh.resolve().as_posix())
            else:
                missing_meshes.append(local_mesh.name)

    # The robot remains usable for FK without its STL meshes.  This fallback
    # prevents a parsing error if someone has not downloaded them yet.
    if missing_meshes:
        for link in root.findall("link"):
            for element in list(link):
                if element.tag in {"visual", "collision"}:
                    link.remove(element)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(
        str(OUTPUT_URDF), encoding="utf-8", xml_declaration=True
    )
    return OUTPUT_URDF


if __name__ == "__main__":
    output_path = prepare_urdf()
    print(f"Created PyBullet-compatible URDF: {output_path}")
