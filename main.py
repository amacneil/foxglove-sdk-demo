import foxglove as fg
from foxglove.schemas import (
    Pose,
    PoseInFrame,
    Quaternion,
    Timestamp,
    Vector3,
    FrameTransform,
    SceneUpdate,
    SceneEntity,
    CompressedImage,
    CubePrimitive,
    Color,
    Duration,
    Grid,
    Vector2,
    PackedElementField,
    PackedElementFieldNumericType,
)
from foxglove.websocket import ServerListener, Client, ChannelView, Capability
import time
from watchfiles import run_process
import numpy as np
import logging
import json
from math import sin, cos, pi


class TeleopListener(ServerListener):
    def __init__(self):
        self.ego_position = {"x": 0, "y": 0, "z": 0}
        self.ego_orientation = {"x": 0, "y": 0, "z": 0, "w": 1}
        self.last_update = time.time()

    def on_message_data(
        self, client: Client, client_channel_id: int, data: bytes
    ) -> None:
        """Handle messages from the client."""
        try:
            msg = json.loads(data)
            print(f"Teleop: {msg}")

            # Update position based on linear velocity
            linear = msg.get("linear", {})
            dx = linear.get("x", 0) * 0.1  # Scale down to 0.1 units
            dy = linear.get("y", 0) * 0.1  # Scale down to 0.1 units
            dz = linear.get("z", 0) * 0.1  # Scale down to 0.1 units

            # Update position in world frame
            self.ego_position["x"] += dx
            self.ego_position["y"] += dy
            self.ego_position["z"] += dz

            # Handle angular velocity for rotation
            angular = msg.get("angular", {})
            if angular.get("z", 0) != 0:
                # Create a rotation quaternion for 15 degrees around Z axis
                angle = pi / 12  # 15 degrees in radians
                if angular["z"] < 0:  # If negative, rotate the other way
                    angle = -angle

                # Create rotation quaternion
                rot_z = sin(angle / 2)
                rot_w = cos(angle / 2)

                # Multiply quaternions (combine rotations)
                new_w = (
                    rot_w * self.ego_orientation["w"]
                    - rot_z * self.ego_orientation["z"]
                )
                new_z = (
                    rot_w * self.ego_orientation["z"]
                    + rot_z * self.ego_orientation["w"]
                )

                # Update orientation dictionary
                self.ego_orientation["z"] = new_z
                self.ego_orientation["w"] = new_w

        except json.JSONDecodeError:
            print(f"Failed to decode message: {data!r}")
        except Exception as e:
            print(f"Error in on_message_data: {e}")
            import traceback

            traceback.print_exc()


def create_ground():
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


def create_landscape():
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


def game_loop(listener):
    # Game loop settings
    TARGET_FPS = 30  # Changed to 30Hz for transforms
    MAP_SCENE_FPS = 1  # Keep map and scene at 1Hz
    FRAME_TIME = 1.0 / TARGET_FPS
    MAP_SCENE_INTERVAL = 1.0 / MAP_SCENE_FPS

    # Create initial landscape and ground
    landscape = create_landscape()
    ground = create_ground()

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

            # Publish ego pose in ego frame at 30Hz
            fg.log(
                "/pose",
                PoseInFrame(
                    timestamp=Timestamp(sec=0),
                    frame_id="ego",
                    pose=Pose(
                        position=Vector3(x=0, y=0, z=0.1),  # 0.1 units above ground
                        orientation=Quaternion(x=0, y=0, z=0, w=1),
                    ),
                ),
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


def main():
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
