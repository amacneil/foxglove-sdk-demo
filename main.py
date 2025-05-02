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
    CubePrimitive,
    Color,
    Duration,
    Grid,
    Vector2,
    PackedElementField,
    PackedElementFieldNumericType,
)
import time
from watchfiles import run_process
import numpy as np


def create_ground():
    # Create a grid for the ground
    width = 11  # -5 to +5
    height = 11  # -5 to +5
    cell_size = 1.0

    # Create a grid with grass color
    data = np.zeros((height, width, 4), dtype=np.uint8)
    data[:, :, 0] = 51  # R: 0.2
    data[:, :, 1] = 204  # G: 0.8
    data[:, :, 2] = 51  # B: 0.2
    data[:, :, 3] = 255  # A: 1.0

    # Create the grid
    ground = Grid(
        timestamp=Timestamp(sec=0),
        frame_id="world",
        pose=Pose(
            position=Vector3(x=-5, y=-5, z=-0.5),  # Start at -5,-5
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
    for z in range(0, 3):
        cubes.append(
            CubePrimitive(
                pose=Pose(
                    position=Vector3(x=2, y=2, z=z),
                    orientation=Quaternion(x=0, y=0, z=0, w=1),
                ),
                size=Vector3(x=0.3, y=0.3, z=1),
                color=Color(r=0.4, g=0.2, b=0.0, a=1.0),  # Wood brown
            )
        )

    # Tree leaves - more compact and tree-like
    # Bottom layer
    for x in range(1, 4):
        for y in range(1, 4):
            cubes.append(
                CubePrimitive(
                    pose=Pose(
                        position=Vector3(x=x, y=y, z=2),
                        orientation=Quaternion(x=0, y=0, z=0, w=1),
                    ),
                    size=Vector3(x=0.3, y=0.3, z=0.3),
                    color=Color(r=0.0, g=0.5, b=0.0, a=1.0),  # Dark green
                )
            )

    # Middle layer - slightly smaller
    for x in range(1, 4):
        for y in range(1, 4):
            cubes.append(
                CubePrimitive(
                    pose=Pose(
                        position=Vector3(x=x, y=y, z=2.3),
                        orientation=Quaternion(x=0, y=0, z=0, w=1),
                    ),
                    size=Vector3(x=0.3, y=0.3, z=0.3),
                    color=Color(r=0.0, g=0.5, b=0.0, a=1.0),  # Dark green
                )
            )

    # Top layer - even smaller
    for x in range(1, 4):
        for y in range(1, 4):
            cubes.append(
                CubePrimitive(
                    pose=Pose(
                        position=Vector3(x=x, y=y, z=2.6),
                        orientation=Quaternion(x=0, y=0, z=0, w=1),
                    ),
                    size=Vector3(x=0.3, y=0.3, z=0.3),
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


def game_loop():
    # Game loop settings
    TARGET_FPS = 1  # Changed to 1Hz
    FRAME_TIME = 1.0 / TARGET_FPS

    # Create initial landscape and ground
    landscape = create_landscape()
    ground = create_ground()

    last_time = time.perf_counter()
    while True:
        current_time = time.perf_counter()
        elapsed = current_time - last_time

        if elapsed >= FRAME_TIME:
            # Publish scene update for landscape
            fg.log("/scene", SceneUpdate(entities=[landscape]))

            # Publish ground grid
            fg.log("/map", ground)

            # Publish transform from world to ego frame
            fg.log(
                "/tf",
                FrameTransform(
                    timestamp=Timestamp(sec=0),
                    parent_frame_id="world",
                    child_frame_id="ego",
                    translation=Vector3(x=1.0, y=2.0, z=0.0),  # Same as ego position
                    rotation=Quaternion(x=0, y=0, z=0, w=1),  # No rotation
                ),
            )

            # Publish ego pose in ego frame
            fg.log(
                "/pose",
                PoseInFrame(
                    timestamp=Timestamp(sec=0),
                    frame_id="ego",
                    pose=Pose(
                        position=Vector3(
                            x=0, y=0, z=0
                        ),  # Position relative to ego frame
                        orientation=Quaternion(x=0, y=0, z=0, w=1),  # No rotation
                    ),
                ),
            )

            last_time = current_time
        else:
            time.sleep(0.001)  # 1ms sleep


def main():
    try:
        # Start the Foxglove websocket server
        server = fg.start_server()
        print(f"Server listening on ws://localhost:{server.port}")

        # Start the game loop
        game_loop()
    except KeyboardInterrupt:
        # Silently exit on keyboard interrupt
        print("Stopped via KeyboardInterrupt")


if __name__ == "__main__":
    run_process(".", target=main)
