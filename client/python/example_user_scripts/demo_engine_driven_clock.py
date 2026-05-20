"""
Demo: engine-driven simulation clock (Unreal host loop).

With ``clock.type`` set to ``engine-driven``, the sim does not use the internal
scheduler; Unreal's game thread supplies frame delta time and the sim consumes it
in fixed steps of ``step-ns`` nanoseconds.

Prerequisites:
  - Project AirSim sim (e.g. Unreal Blocks) running and reachable by the client.
  - This script loads ``scene_engine_driven_demo.jsonc`` from the ``sim_config``
    folder next to the standard example layouts.
"""

import asyncio

import sys
import time
from pathlib import Path

from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.image_utils import ImageDisplay
from projectairsim.utils import projectairsim_log


SCENE = "scene_engine_driven_demo.jsonc"
DRONE_NAME = "Drone1"
STEP_NS = 3_000_000  # must match scene_engine_driven_demo.jsonc
SAMPLES = 6
SAMPLE_INTERVAL_SEC = 0.5


async def main() -> int:
    script_dir = Path(__file__).resolve().parent
    sim_config_dir = script_dir / "sim_config"

    client = ProjectAirSimClient()
    world: World | None = None

    try:
        print("Connecting to Project AirSim…")
        client.connect()

        print(f"Loading scene '{SCENE}' (engine-driven clock, step-ns={STEP_NS})…")
        world = World(
            client,
            SCENE,
            delay_after_load_sec=1.0,
            sim_config_path=str(sim_config_dir),
        )

        clock_type = world.get_sim_clock_type()
        print(f"Sim clock type: {clock_type}")
        if "engine-driven" not in clock_type.lower():
            print(
                "Warning: expected an engine-driven clock for this demo scene.",
                file=sys.stderr,
            )

        t0 = world.get_sim_time()
        print(f"Initial sim time: {t0 * 1e-9:.6f} s")

        print(
            f"Sampling sim time every {SAMPLE_INTERVAL_SEC}s "
            f"({SAMPLES} samples). Advancement is driven by the Unreal frame loop."
        )
        prev = t0
        for i in range(1, SAMPLES + 1):
            time.sleep(SAMPLE_INTERVAL_SEC)
            t = world.get_sim_time()
            delta_ns = t - prev
            print(
                f"  [{i}] sim_time={t * 1e-9:.6f} s  "
                f"(delta since last sample: {delta_ns * 1e-6:.3f} ms)"
            )
            prev = t

        drone = Drone(client, world, DRONE_NAME)
        drone.enable_api_control()
        drone.arm()

        # Subscribe to chase camera sensor as a client-side pop-up window
        chase_cam_window = "ChaseCam"
        image_display = ImageDisplay()
        image_display.add_chase_cam(chase_cam_window)
        client.subscribe(
            drone.sensors["Chase"]["scene_camera"],
            lambda _, chase: image_display.receive(chase, chase_cam_window),
        )

        # Subscribe to the downward-facing camera sensor's RGB and Depth images
        rgb_name = "RGB-Image"
        image_display.add_image(rgb_name, subwin_idx=0)
        client.subscribe(
            drone.sensors["DownCamera"]["scene_camera"],
            lambda _, rgb: image_display.receive(rgb, rgb_name),
        )

        depth_name = "Depth-Image"
        image_display.add_image(depth_name, subwin_idx=2)
        client.subscribe(
            drone.sensors["DownCamera"]["depth_camera"],
            lambda _, depth: image_display.receive(depth, depth_name),
        )

        image_display.start()

        # ------------------------------------------------------------------------------

        # Set the drone to be ready to fly
        drone.enable_api_control()
        drone.arm()

        pose = drone.get_ground_truth_pose()
        tr = pose["translation"]
        print(
            f"Drone '{DRONE_NAME}' ground-truth position (NED): "
            f"north={tr['x']:.3f} east={tr['y']:.3f} down={tr['z']:.3f} m"
        )

        # Command the drone to move up in NED coordinate system at 1 m/s for 4 seconds
        move_up_task = await drone.move_by_velocity_async(
            v_north=0.0, v_east=0.0, v_down=-1.0, duration=4.0
        )
        projectairsim_log().info("Move-Up invoked")

        move_up_task
        projectairsim_log().info("Move-Up completed")

        # ------------------------------------------------------------------------------

        # Command the Drone to move down in NED coordinate system at 1 m/s for 4 seconds
        move_down_task = await drone.move_by_velocity_async(
            v_north=0.0, v_east=0.0, v_down=1.0, duration=4.0
        )  # schedule an async task to start the command
        projectairsim_log().info("Move-Down invoked")

        # Example 2: Wait for move_down_task to complete before continuing
        while not move_down_task.done():
            await asyncio.sleep(0.005)
        projectairsim_log().info("Move-Down completed")

        # ------------------------------------------------------------------------------

        projectairsim_log().info("land_async: starting")
        land_task = await drone.land_async()
        await land_task
        projectairsim_log().info("land_async: completed")

        # ------------------------------------------------------------------------------

        # Shut down the drone
        drone.disarm()
        drone.disable_api_control()

        print("Done.")
        return 0
    except Exception as exc:
        print(f"Demo failed: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            client.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())  # Runner for async main function