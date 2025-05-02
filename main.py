import foxglove as fg
from foxglove.schemas import Pose, PoseInFrame, Quaternion, Timestamp, Vector3
import time
from watchfiles import run_process


def game_loop():
    # Game loop settings
    TARGET_FPS = 10
    FRAME_TIME = 1.0 / TARGET_FPS

    last_time = time.perf_counter()
    while True:
        current_time = time.perf_counter()
        elapsed = current_time - last_time

        if elapsed >= FRAME_TIME:
            # Example: Log a pose to a topic
            fg.log(
                "/example_pose",
                PoseInFrame(
                    timestamp=Timestamp(sec=0),
                    frame_id="world",
                    pose=Pose(
                        position=Vector3(x=0, y=0, z=0),
                        orientation=Quaternion(x=0, y=0, z=0, w=1),
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
