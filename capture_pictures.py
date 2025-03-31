"""
Script to capture pictures to be used for background removal.
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np
import odrive
from icrpy_utils import odriveapi, vision

"""
↓ Parameters
"""

# position the robot will go to to take the picture
bx, by = 5, 5

"""
↑ Parameters
"""

OK_VALUES = ["y", "Y", "yes", "Yes", "o", "O", "oui", "Oui"]


def main(
    _cap: Optional[Tuple[cv2.VideoCapture, np.ndarray]] = None,
    odrv: Optional[odriveapi.OdriveAPI] = None,
) -> int:
    # if not already created, create video capture object
    if _cap is None:
        _cap = vision.cap()

    cap, im_shape = _cap

    # if not already setup, setup the odrive
    if odrv is None:
        # find the odrive
        _odrv = odrive.find_any()
        _odrv.axis0.controller.config.vel_limit = 5
        _odrv.axis1.controller.config.vel_limit = 5
        _odrv.axis0.controller.config.vel_limit_tolerance = 500000
        _odrv.axis1.controller.config.vel_limit_tolerance = 500000

        odrv = odriveapi.OdriveAPI(_odrv)
        odrv.startup()

    odrv._odrv.clear_errors()
    time.sleep(0.1)
    odrv._odrv.clear_errors()
    time.sleep(0.1)

    stop = False
    while not stop:
        print("moving to correct spot")

        odrv.cartesian_move(bx, by, 3)

        name: str
        ok = False
        while not ok:
            name = input("Enter file name (no extension) : ")
            print(f"entered name: '{name}'")
            ok = input("is this ok ?") in OK_VALUES

        print("capturing picture")
        ret, frame = cap.read()
        if not ret:
            print("error capturing frame. exiting")
            break

        cv2.imwrite(f"{name}.png", frame)

        # enter any value that is not in OK_VALUES to exit
        # most commonly, use y to continue and n to exit
        stop = input("continue ? (y/n)") not in OK_VALUES

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
