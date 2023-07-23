from cozmo_fsm import *

import os
import glob
import torch
import cv2  # f
import argparse
import time
import numpy as np

import sys

sys.path.append("./MiDaS/")
sys.path.append("./MiDaS/midas/")
from midas.model_loader import load_model
import re
import numpy as np
import cv2
import torch

import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: %s" % device)
model, transform, net_w, net_h = load_model(
    device=device,
    model_path="MiDaS/weights/dpt_swin2_tiny_256.pt",
    model_type="dpt_swin2_tiny_256",
    optimize=False,
)


# function to return the distance to any location on the image via an onclick
def onclick(event, prediction):
    ix, iy = event.xdata, event.ydata
    # print(ix, iy)
    if prediction is not None:
        print(
            f"distance to the location you clicked is {round(prediction[int(round(iy, 0)), int(round(ix, 0))] / 10, 3)} cm"
        )
    else:
        print("no cube available for normalization")

    return None


def write_depth(depth, grayscale=False, bits=1):
    """Write depth map to png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
        grayscale (bool): use a grayscale colormap?
    """
    if not grayscale:
        bits = 1

    if not np.isfinite(depth).all():
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2 ** (8 * bits)) - 1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_PARULA)

    if bits == 1:
        return out.astype("uint8")
    elif bits == 2:
        return out.astype("uint16")
    return


class Depth(StateMachineProgram):
    def start(self):
        super().start()
        print("Type 'tm' to begin the depth process")

    class DisplayCamera(StateNode):
        def start(self, event=None):
            super().start(event)
            old_image = (
                self.robot.world.latest_image.raw_image
            )  # get the image from the camera world
            image_boxes = []  # list to store all pixel-locations of cubes
            cubes = []  # list to store which cubes we are looking at

            # fill image boxes and cubes list
            image_box_1 = cube1.last_observed_image_box
            if image_box_1 is not None:
                print(f"cozmo sees cube 1, calibrating location")
                image_boxes.append(image_box_1)
                cubes.append(1)

            image_box_2 = cube2.last_observed_image_box
            if image_box_2 is not None:
                print(f"cozmo sees cube 2, calibrating location")
                image_boxes.append(image_box_2)
                cubes.append(2)
            image_box_3 = cube3.last_observed_image_box
            if image_box_3 is not None:
                print(f"cozmo sees cube 3, calibrating location")
                image_boxes.append(image_box_3)
                cubes.append(3)

            # we will use the bottom center of the cube as our pixel location of the cube
            image_box_vals = []
            for image_box in image_boxes:
                middlex = image_box.top_left_x + image_box.width / 2
                lowy = np.array(old_image).shape[0] - (image_box.top_left_y + image_box.height)
                image_box_vals.append((middlex, lowy))

            # update image to be passed into tensor
            original_image_rgb = cv2.cvtColor(np.array(old_image), cv2.COLOR_BGR2RGB) / 255.0

            # transform the image according to the model specifications (for us, this corresponds to resizing the image to go into the model)
            image = transform({"image": original_image_rgb})["image"]
            # generate depth information
            with torch.no_grad():
                sample = torch.from_numpy(image).to(device).unsqueeze(0)
                prediction = model.forward(sample)
                prediction = (
                    torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=original_image_rgb.shape[1::-1][::-1],
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )

            # use kinematics to find distances to all of the cubes
            distances_to_cubes = []
            for cubeseen in cubes:
                if cubeseen == 1:
                    pt = geometry.point(wcube1.x, wcube1.y, 0)
                elif cubeseen == 2:
                    pt = geometry.point(wcube2.x, wcube2.y, 0)
                elif cubeseen == 3:
                    pt = geometry.point(wcube3.x, wcube3.y, 0)

                translated_point = robot.kine.base_to_joint("camera").dot(pt)
                distance = (
                    translated_point[0][0] ** 2
                    + translated_point[1][0] ** 2
                    + translated_point[2][0] ** 2
                ) ** 0.5
                distances_to_cubes.append(distance)
                print(f"distance to cube {cubeseen} is {round(distance / 10, 2)} cm")
            # update NaN values
            if not np.isfinite(prediction).all():
                depth = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
                print("WARNING: Non-finite depth values present")

            depth_min = prediction.min()
            depth_max = prediction.max()
            # normalize depth so that it is between 0 and 1
            if depth_max - depth_min > np.finfo("float").eps:
                prediction = (prediction - depth_min) / (depth_max - depth_min)

            # remove zero values, then flip depth prediction so that guesses of farther away objects have higher scores (I'm not sure why the model produces it the other way around)
            prediction = prediction + 1e-8
            prediction_inverted = 1 / prediction
            prediction_normalized = None

            cube_depth_estimates = (
                []
            )  # compute current depth estimates at all locations of cubes (note these values are roughly between 0 and 1)
            for middlex, lowy in image_box_vals:
                cube_depth_estimates.append(
                    prediction_inverted[
                        int(round(prediction.shape[0] - lowy, 0)), int(round(middlex, 0))
                    ]
                )

            # compute mathematically optimal scaling factor
            dot_product = sum(
                [
                    distances_to_cubes[x] * cube_depth_estimates[x]
                    for x in range(len(cube_depth_estimates))
                ]
            )
            magnitude = sum(
                [cube_depth_estimates[x] ** 2 for x in range(len(cube_depth_estimates))]
            )
            if magnitude != 0:
                scaling_factor = dot_product / magnitude
            else:
                scaling_factor = 1

            prediction_normalized = (
                prediction_inverted * scaling_factor
            )  # compute actual depth guesses

            # prediction can be used for whatever task
            # display the pixel location of the cube depths on the matplotlib screen
            for middlex, lowy in image_box_vals:
                for x in range(7):
                    for y in range(7):
                        prediction[
                            prediction.shape[0] - int(round(lowy, 0)) + 3 - y,
                            int(round(middlex, 0)) + 3 - x,
                        ] = 0

            depth = write_depth(prediction)
            cv2.imwrite("im.png", depth)
            fig = plt.figure(1)
            # add functionality so that a click from a user results in depth prediction
            cid = fig.canvas.mpl_connect(
                "button_press_event", lambda x: onclick(x, prediction_normalized)
            )
            plt.imshow(depth)

            plt.pause(0.01)

    def setup(self):

        # Code generated by genfsm on Thu May  4 20:07:25 2023:

        start = StateNode().set_name("start").set_parent(self)
        display = self.DisplayCamera().set_name("display").set_parent(self)

        textmsgtrans1 = TextMsgTrans().set_name("textmsgtrans1")
        textmsgtrans1.add_sources(start).add_destinations(display)

        timertrans1 = TimerTrans(0.1).set_name("timertrans1")
        timertrans1.add_sources(display).add_destinations(display)

        return self
