import json
import time
from math import cos, pi, sin

import foxglove as fg
import numpy as np
from foxglove.schemas import (
    Color,
    CubePrimitive,
    Duration,
    FrameTransform,
    Grid,
    LinePrimitive,
    PackedElementField,
    PackedElementFieldNumericType,
    Pose,
    Quaternion,
    SceneEntity,
    SceneUpdate,
    Timestamp,
    Vector2,
    Vector3,
)
from foxglove.websocket import Capability, Client, ServerListener
from scipy.spatial.transform import Rotation
from watchfiles import run_process


class TeleopListener(ServerListener):
    def __init__(self) -> None:
        self.ego_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.ego_orientation = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        self.last_update = time.time()

    def apply_rotation_to_vector(self, q: dict, v: dict) -> dict:
        """Apply quaternion rotation to a vector using scipy.

        Args:
            q: Quaternion dictionary with x, y, z, w keys
            v: Vector dictionary with x, y, z keys

        Returns:
            Rotated vector dictionary with x, y, z keys
        """
        # Create a Rotation object directly - scipy uses [x, y, z, w]
        r = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]])

        # Create vector
        vec = np.array([v["x"], v["y"], v["z"]])

        # Apply rotation
        rotated_vec = r.apply(vec)

        # Return as dictionary
        return {
            "x": float(rotated_vec[0]),
            "y": float(rotated_vec[1]),
            "z": float(rotated_vec[2]),
        }

    def on_message_data(
        self, client: Client, client_channel_id: int, data: bytes
    ) -> None:
        """Handle messages from the client."""
        try:
            msg = json.loads(data)
            print(f"Teleop: {msg}")

            # Extract linear velocities
            linear = msg.get("linear", {})
            linear_vec = {
                "x": linear.get("x", 0) * 0.1,  # Scale down to 0.1 units
                "y": linear.get("y", 0) * 0.1,  # Scale down to 0.1 units
                "z": linear.get("z", 0) * 0.1,  # Scale down to 0.1 units
            }

            # Apply rotation to linear velocity vector based on current orientation
            rotated_vec = self.apply_rotation_to_vector(
                self.ego_orientation, linear_vec
            )

            # Update position in world frame with rotated velocity
            self.ego_position["x"] += rotated_vec["x"]
            self.ego_position["y"] += rotated_vec["y"]
            self.ego_position["z"] += rotated_vec["z"]

            # Handle angular velocity for rotation
            angular = msg.get("angular", {})
            if angular.get("z", 0) != 0:
                # Create a rotation quaternion for 7.5 degrees around Z axis
                angle = pi / 24  # 7.5 degrees in radians
                if angular["z"] < 0:  # If negative, rotate the other way
                    angle = -angle

                # Create current rotation directly
                current_rot = Rotation.from_quat(
                    [
                        self.ego_orientation["x"],
                        self.ego_orientation["y"],
                        self.ego_orientation["z"],
                        self.ego_orientation["w"],
                    ]
                )

                # Create rotation for the new angle around Z
                delta_rot = Rotation.from_rotvec([0, 0, angle])

                # Compose rotations
                new_rot = delta_rot * current_rot

                # Get new quaternion [x, y, z, w]
                new_quat = new_rot.as_quat()

                # Update orientation dictionary
                self.ego_orientation["x"] = float(new_quat[0])
                self.ego_orientation["y"] = float(new_quat[1])
                self.ego_orientation["z"] = float(new_quat[2])
                self.ego_orientation["w"] = float(new_quat[3])

        except json.JSONDecodeError:
            print(f"Failed to decode message: {data!r}")
        except Exception as e:
            print(f"Error in on_message_data: {e}")
            import traceback

            traceback.print_exc()


def create_ground() -> Grid:
    # Create a grid for the ground
    width = 11  # -5 to +5
    height = 11  # -5 to +5
    cell_size = 1.0

    # Create a grid with grass color
    data = np.zeros((height, width, 4), dtype=np.uint8)
    data[:, :, 0] = 0  # R: 0.0
    data[:, :, 1] = 100  # G: 0.4
    data[:, :, 2] = 0  # B: 0.0
    data[:, :, 3] = 255  # A: 1.0

    # Create the grid
    ground = Grid(
        timestamp=Timestamp(sec=0),
        frame_id="world",
        pose=Pose(
            position=Vector3(x=-5, y=-5, z=0),  # Changed to z=0
            orientation=Quaternion(x=0, y=0, z=0, w=1),
        ),
        column_count=width,
        cell_size=Vector2(x=cell_size, y=cell_size),
        row_stride=width * 4,  # 4 bytes per pixel (RGBA)
        cell_stride=4,  # 4 bytes per cell (RGBA)
        fields=[
            PackedElementField(
                name="red", offset=0, type=PackedElementFieldNumericType.Uint8
            ),
            PackedElementField(
                name="green", offset=1, type=PackedElementFieldNumericType.Uint8
            ),
            PackedElementField(
                name="blue", offset=2, type=PackedElementFieldNumericType.Uint8
            ),
            PackedElementField(
                name="alpha", offset=3, type=PackedElementFieldNumericType.Uint8
            ),
        ],
        data=data.tobytes(),
    )

    return ground


def create_landscape() -> SceneEntity:
    # Create cubes for the landscape
    cubes = []

    # Add some decorative blocks (trees, etc.)
    # Tree trunks
    for z in range(0, 2):  # 2 blocks high
        cubes.append(
            CubePrimitive(
                pose=Pose(
                    position=Vector3(x=2, y=2, z=z + 0.5),  # Start at z=0.5 and go up
                    orientation=Quaternion(x=0, y=0, z=0, w=1),
                ),
                size=Vector3(x=0.3, y=0.3, z=1),
                color=Color(r=0.4, g=0.2, b=0.0, a=1.0),  # Wood brown
            )
        )

    # Single cube for leaves, wider than the trunk
    cubes.append(
        CubePrimitive(
            pose=Pose(
                position=Vector3(x=2, y=2, z=2.0),  # Directly above trunk
                orientation=Quaternion(x=0, y=0, z=0, w=1),
            ),
            size=Vector3(
                x=0.9, y=0.9, z=0.9
            ),  # 3x wider than trunk and same height as width
            color=Color(r=0.0, g=0.5, b=0.0, a=1.0),  # Dark green
        )
    )

    # Create a scene entity for the landscape
    landscape = SceneEntity(
        timestamp=Timestamp(sec=0),
        frame_id="world",
        id="landscape",
        lifetime=Duration(sec=0, nsec=0),  # Never expire
        frame_locked=False,
        metadata=[],
        cubes=cubes,
    )

    return landscape


def create_robot() -> SceneEntity:
    """Create a cute gray robot using cubes."""
    cubes = []
    lines: list[LinePrimitive] = []

    # Robot body (gray)
    gray_color = Color(r=0.7, g=0.7, b=0.7, a=1.0)
    dark_gray_color = Color(r=0.5, g=0.5, b=0.5, a=1.0)
    black_color = Color(r=0.1, g=0.1, b=0.1, a=1.0)

    # Robot body
    cubes.append(
        CubePrimitive(
            pose=Pose(
                position=Vector3(x=0, y=0, z=0.4),  # Main body
                orientation=Quaternion(x=0, y=0, z=0, w=1),
            ),
            size=Vector3(x=0.5, y=0.4, z=0.4),
            color=gray_color,
        )
    )

    # Robot head
    cubes.append(
        CubePrimitive(
            pose=Pose(
                position=Vector3(x=0, y=0, z=0.8),  # Head above body
                orientation=Quaternion(x=0, y=0, z=0, w=1),
            ),
            size=Vector3(x=0.35, y=0.35, z=0.3),
            color=gray_color,
        )
    )

    # Robot arms
    # Left upper arm
    cubes.append(
        CubePrimitive(
            pose=Pose(
                position=Vector3(x=0, y=0.25, z=0.45),
                orientation=Quaternion(x=0, y=0, z=0.38, w=0.92),  # ~45 degrees
            ),
            size=Vector3(x=0.1, y=0.25, z=0.1),
            color=gray_color,
        )
    )

    # Left forearm
    cubes.append(
        CubePrimitive(
            pose=Pose(
                position=Vector3(x=0.04, y=0.40, z=0.45),
                orientation=Quaternion(x=0, y=0, z=0, w=1),
            ),
            size=Vector3(x=0.1, y=0.2, z=0.1),
            color=gray_color,
        )
    )

    # Left hand
    cubes.append(
        CubePrimitive(
            pose=Pose(
                position=Vector3(x=0.04, y=0.52, z=0.45),
                orientation=Quaternion(x=0, y=0, z=0, w=1),
            ),
            size=Vector3(x=0.12, y=0.08, z=0.12),
            color=dark_gray_color,
        )
    )

    # Right upper arm
    cubes.append(
        CubePrimitive(
            pose=Pose(
                position=Vector3(x=0, y=-0.25, z=0.45),
                orientation=Quaternion(x=0, y=0, z=-0.38, w=0.92),  # ~-45 degrees
            ),
            size=Vector3(x=0.1, y=0.25, z=0.1),
            color=gray_color,
        )
    )

    # Right forearm
    cubes.append(
        CubePrimitive(
            pose=Pose(
                position=Vector3(x=0.04, y=-0.40, z=0.45),
                orientation=Quaternion(x=0, y=0, z=0, w=1),
            ),
            size=Vector3(x=0.1, y=0.2, z=0.1),
            color=gray_color,
        )
    )

    # Right hand
    cubes.append(
        CubePrimitive(
            pose=Pose(
                position=Vector3(x=0.04, y=-0.52, z=0.45),
                orientation=Quaternion(x=0, y=0, z=0, w=1),
            ),
            size=Vector3(x=0.12, y=0.08, z=0.12),
            color=dark_gray_color,
        )
    )

    # Robot wheels/tracks
    cubes.append(
        CubePrimitive(
            pose=Pose(
                position=Vector3(x=0, y=-0.25, z=0.15),  # Right wheel
                orientation=Quaternion(x=0, y=0, z=0, w=1),
            ),
            size=Vector3(x=0.45, y=0.1, z=0.3),
            color=dark_gray_color,
        )
    )

    cubes.append(
        CubePrimitive(
            pose=Pose(
                position=Vector3(x=0, y=0.25, z=0.15),  # Left wheel
                orientation=Quaternion(x=0, y=0, z=0, w=1),
            ),
            size=Vector3(x=0.45, y=0.1, z=0.3),
            color=dark_gray_color,
        )
    )

    # Create a scene entity for the robot
    robot = SceneEntity(
        timestamp=Timestamp(sec=0),
        frame_id="ego",
        id="robot",
        lifetime=Duration(sec=0, nsec=0),  # Never expire
        frame_locked=True,  # This will follow the ego frame
        metadata=[],
        cubes=cubes,
        lines=lines,
    )

    return robot


def game_loop(listener: TeleopListener) -> None:
    # Game loop settings
    TARGET_FPS = 30  # Changed to 30Hz for transforms
    MAP_SCENE_FPS = 1  # Keep map and scene at 1Hz
    FRAME_TIME = 1.0 / TARGET_FPS
    MAP_SCENE_INTERVAL = 1.0 / MAP_SCENE_FPS

    # Create initial landscape and ground
    landscape = create_landscape()
    ground = create_ground()
    robot = create_robot()

    last_time = time.perf_counter()
    last_map_scene_time = last_time

    while True:
        current_time = time.perf_counter()
        elapsed = current_time - last_time
        map_scene_elapsed = current_time - last_map_scene_time

        if elapsed >= FRAME_TIME:
            # Always publish transform from world to ego frame at 30Hz
            fg.log(
                "/tf",
                FrameTransform(
                    timestamp=Timestamp(sec=0),
                    parent_frame_id="world",
                    child_frame_id="ego",
                    translation=Vector3(
                        x=listener.ego_position["x"],
                        y=listener.ego_position["y"],
                        z=listener.ego_position["z"],
                    ),
                    rotation=Quaternion(
                        x=listener.ego_orientation["x"],
                        y=listener.ego_orientation["y"],
                        z=listener.ego_orientation["z"],
                        w=listener.ego_orientation["w"],
                    ),
                ),
            )

            # Publish robot entity instead of pose
            fg.log(
                "/ego",
                SceneUpdate(entities=[robot]),
            )

            # Update map and scene at 1Hz
            if map_scene_elapsed >= MAP_SCENE_INTERVAL:
                # Publish scene update for landscape
                fg.log("/scene", SceneUpdate(entities=[landscape]))

                # Publish ground grid
                fg.log("/map", ground)

                last_map_scene_time = current_time

            last_time = current_time
        else:
            time.sleep(0.001)  # 1ms sleep


def main() -> None:
    try:
        # Create our listener
        listener = TeleopListener()

        # Start the Foxglove websocket server with our listener
        server = fg.start_server(
            server_listener=listener,
            capabilities=[Capability.ClientPublish],
            supported_encodings=["json"],
        )
        print(f"Server listening on ws://localhost:{server.port}")

        # Start the game loop
        game_loop(listener)
    except KeyboardInterrupt:
        # Silently exit on keyboard interrupt
        print("Stopped via KeyboardInterrupt")


if __name__ == "__main__":
    run_process(".", target=main)
