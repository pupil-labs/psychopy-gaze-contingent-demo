import time
import random
import math
import csv

import numpy as np
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, iohub, hardware

from pupil_labs.realtime_api.simple import discover_one_device
from pupil_labs.realtime_screen_gaze import (
    marker_generator,
    cloud_api,
)
from pupil_labs.realtime_screen_gaze.gaze_mapper import GazeMapper


# Settings
marker_ids = [0, 1, 2, 3]
target_radius = 100         # pixels
interstimulus_interval = 1  # seconds
marker_size = 200
marker_opacity = 0.75
visible_gaze = False


class MarkerStim(visual.ImageStim):
    def __init__(self, marker_id, *args, **kwargs):
        self.marker_id = marker_id

        marker_data = marker_generator.generate_marker(marker_id, flip_x=True).astype(float)
        marker_data[marker_data == 0] = -1
        marker_data[marker_data > 0] = 1

        marker_data = np.pad(marker_data, pad_width=1, mode="constant", constant_values=1)
        super().__init__(image=marker_data, *args, **kwargs)

    def get_marker_verts(self):
        padding = self.size[0]/10

        topLeft = (
            self.pos[0] + padding,
            self.pos[1] + self.size[1] - padding
        )
        bottomRight = (
            self.pos[0] + self.size[0] - padding,
            self.pos[1] + padding
        )

        return (
            topLeft,
            (bottomRight[0], topLeft[1]),
            bottomRight,
            (topLeft[0], bottomRight[1]),
        )


def main():
    window = visual.Window(fullscr=True, screen=0, allowStencil=False, units="pix")

    status_label = visual.TextStim(win=window, text="", height=32)
    status_label.autoDraw = True

    # Look for devices. Returns as soon as it has found the first device.
    status_label.text = "Looking for the next best device..."
    window.flip()
    time.sleep(5)

    device = discover_one_device(max_search_duration_seconds=10)
    if device is None:
        status_label.text = "No device found. Exiting..."
        window.flip()

        time.sleep(2)
        raise SystemExit(-1)

    status_label.text = f"Found {str(device)}.\nConnecting..."
    window.flip()


    # Pull scene camera serial to fetch accurate camera intrinsics
    camera_serial = device.module_serial or device.serial_number_scene_cam
    if not camera_serial:
        status_label.text = "Scene camera not connected. Exiting..."
        window.flip()
        device.close()
        time.sleep(2)
        raise SystemExit(-2)

    camera = cloud_api.camera_for_scene_cam_serial(camera_serial)
    gaze_mapper = GazeMapper(camera)


    # Setup screen markers and surface
    marker_padding = marker_size / 8
    markers = [
        MarkerStim(
            marker_id=marker_ids[0],
            win=window, name="marker-0",
            units="pix", size=(marker_size, marker_size), opacity=marker_opacity,
            pos=(
                (-window.size[0] + marker_size) / 2,
                (window.size[1] - marker_size) / 2
            )
        ),
        MarkerStim(
            marker_id=marker_ids[1],
            win=window, name="marker-1",
            units="pix", size=(marker_size, marker_size), opacity=marker_opacity,
            pos=(
                (window.size[0] - marker_size) / 2,
                (window.size[1] - marker_size) / 2
            )
        ),
        MarkerStim(
            marker_id=marker_ids[2],
            win=window, name="marker-2",
            units="pix", size=(marker_size, marker_size), opacity=marker_opacity,
            pos=(
                (window.size[0] - marker_size) / 2,
                (-window.size[1] + marker_size) / 2
            )
        ),
        MarkerStim(
            marker_id=marker_ids[3],
            win=window, name="marker-3",
            units="pix", size=(marker_size, marker_size), opacity=marker_opacity,
            pos=(
                (-window.size[0] + marker_size) / 2,
                (-window.size[1] + marker_size) / 2
            )
        ),
    ]
    marker_verts = {}

    for marker in markers:
        marker.autoDraw = True
        marker_verts[marker.marker_id] = marker.get_marker_verts() + window.size/2

    screen_surface = gaze_mapper.add_surface(
        marker_verts,
        window.size
    )


    # Setup other visual stimuli
    distractors = []
    for distractor_idx in range(10):
        stim = visual.Circle(win=window, name=f"distractor-{distractor_idx}", radius=target_radius, color="darkgray", units="pix")
        distractors.append(stim)

    target = visual.Circle(win=window, name="target", radius=target_radius, color="blue", units="pix")
    stims = distractors + [target]

    gaze_indicator = visual.Circle(win=window, name="gaze", radius=8, color="red", units="pix")

    status_label.text = ""

    # Trial Loop
    for trial_id in range(10):
        for stim in stims:
            stim.pos = (
                (random.random() - 0.5) * (window.size[0] - 2*(target_radius + marker_size)),
                (random.random() - 0.5) * (window.size[1] - 2*(target_radius + marker_size))
            )
            stim.draw()

        target_hit = False
        while not target_hit:
            for stim in distractors + [target]:
                stim.draw()

            # Wait for scene camera image and corresponding gaze position
            frame, gaze = device.receive_matched_scene_video_frame_and_gaze()
            result = gaze_mapper.process_frame(frame, gaze)

            for surface_gaze in result.mapped_gaze[screen_surface.uid]:
                gaze_indicator.pos = (
                    (surface_gaze.x-0.5) * window.size[0],
                    -(surface_gaze.y-0.5) * window.size[1]
                )
                if visible_gaze:
                    gaze_indicator.draw()

                distance = math.sqrt((gaze_indicator.pos[0] - target.pos[0])**2 + (gaze_indicator.pos[1] - target.pos[1])**2)
                if distance < target_radius:
                    target_hit = True
                    break

            window.flip()

        window.flip()
        time.sleep(interstimulus_interval)


    print("Stopping...")
    device.close()

if __name__ == "__main__":
    main()