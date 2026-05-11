# -*- coding: utf-8 -*-
"""
Braccio arm high-level motion control over serial.
Original author: natan (2021)
"""

import serial
import time
import solverNNA
import numpy as np

# Default servo angle limits and index in command array
# Format: [current_angle, min_angle, max_angle, array_index]
base      = [0,  0,  180, 0]
shoulder  = [150, 15, 165, 1]
elbow     = [0,  0,  180, 2]
wrist     = [0,  0,  180, 3]
wristRot  = [90, 0,  180, 4]
gripper   = [73, 73,  0,  5]   # 73 = closed, 0 = open

# Initialize serial connection to Arduino Braccio
# Change 'COM4' to your actual port (e.g. '/dev/ttyUSB0' on Linux/Mac)
arm = serial.Serial('COM12', 115200, timeout=5)
print("Initializing arm")
time.sleep(2)
arm.write(b'H0,90,20,90,90,73,20\n')  # home at low speed
time.sleep(2)


def write_arduino(angles):
    """Invert base and wrist for servo orientation, then send command."""
    angles[0] = 180 - angles[0]
    angles[3] = 180 - angles[3]
    angle_str = ','.join(str(a) for a in angles)
    cmd = f"P{angle_str},200\n"
    arm.write(cmd.encode())


def rotate_joint(joint):
    def move_to(limit_index):
        angles = [base[0], shoulder[0], elbow[0], wrist[0], wristRot[0], gripper[0]]
        angles[joint[3]] = joint[limit_index]
        write_arduino(angles)
    move_to(1); time.sleep(2)
    move_to(2); time.sleep(2)
    move_to(1); time.sleep(2)


def home(speed=20):
    """Move arm to home/rest position."""
    angles = [base[0], shoulder[0], elbow[0], wrist[0], wristRot[0], gripper[0]]
    write_arduino(angles)


def get_previous_teta():
    """Read previously saved joint angles from file."""
    text_file = open("prev_teta.txt", "r")
    prev_teta_string = text_file.read()
    text_file.close()
    prev_teta = list(prev_teta_string.split(";"))
    prev_teta.pop(6)
    prev_teta = [int(i) for i in prev_teta]
    return prev_teta


def write_position(theta_base, theta_shoulder, theta_elbow, theta_wrist,
                   theta_wristRot, grip="closed"):
    """Apply backlash compensation and send joint angles to the arm."""
    theta_gripper = gripper[1] if grip == "closed" else gripper[2]
    tb_comp = solverNNA.backlash_compensation_base(theta_base)
    angles = [tb_comp, theta_shoulder, theta_elbow, theta_wrist, theta_wristRot, theta_gripper]
    write_arduino(angles)
    # Save angles for next compensation step
    with open("prev_teta.txt", "w") as f:
        for a in [theta_base, theta_shoulder, theta_elbow, theta_wrist, theta_wristRot, theta_gripper]:
            f.write(f"{a};")


def go_to_coordinate(x, y, z, grip_position="closed"):
    """Move end-effector to Cartesian (x, y, z) with IK."""
    th = solverNNA.move_to_position_cart(x, y, z)
    write_position(th[0], th[1], th[2], th[3], 90, grip=grip_position)


def open_gripper():
    prev = get_previous_teta()
    write_position(prev[0], prev[1], prev[2], prev[3], prev[4], grip="open")


def close_gripper():
    prev = get_previous_teta()
    write_position(prev[0], prev[1], prev[2], prev[3], prev[4], grip="closed")


def pick_up(x, y):
    """
    Full pick-and-place sequence:
    home -> approach -> open -> descend -> grasp -> lift -> deposit -> release -> home
    """
    glass_pos = [310, 95]   # fixed deposit location
    delay = 1
    pick_height = 10

    home()
    time.sleep(delay)

    go_to_coordinate(x, y, 100, "closed");  time.sleep(delay)  # approach above
    open_gripper();                          time.sleep(delay)  # open gripper
    go_to_coordinate(x, y, pick_height - 20, "open"); time.sleep(delay)  # descend
    close_gripper();                         time.sleep(delay)  # grasp
    go_to_coordinate(x, y, 200, "closed");  time.sleep(delay)  # lift

    go_to_coordinate(glass_pos[0], glass_pos[1], 200, "closed"); time.sleep(delay)  # move over deposit
    go_to_coordinate(glass_pos[0], glass_pos[1], 120, "closed"); time.sleep(delay)  # lower
    open_gripper()                                                                    # release

    home()


def camera_compensation(x_coord, y_coord):
    """
    Correct pixel-space centroid coordinates for perspective foreshortening
    using known camera height and object (foam) height.
    """
    h_foam  = 80                 # foam object height in mm
    cam_pos = [480, 150, 880]    # camera position [x, y, z] in mm
    offset  = 300

    # Reposition x relative to workspace centre
    x_adj = (offset - x_coord) + (cam_pos[0] - offset)

    # Perspective correction via similar triangles
    x_corr = x_adj - (h_foam / (cam_pos[2] / x_adj))

    if y_coord < cam_pos[1]:
        y_corr = y_coord - (h_foam / (cam_pos[2] / y_coord))
    else:
        y_corr = y_coord + (h_foam / (cam_pos[2] / y_coord))

    x_final = offset - (x_corr - (cam_pos[0] - offset))
    return int(x_final), int(y_corr)

def pick_and_place_cartesian(x_pick, y_pick):
    """
    Sends Cartesian coordinates directly to the Arduino.
    The Arduino will handle the IK and the full motion sequence.
    """
    # Pick height (Z-axis)
    z_pick = 10  
    
    # Hardcoded Drop/Place location (based on your original glass_pos)
    x_place = 310
    y_place = 95
    z_place = 120 
    
    # Format the command: G:x1,y1,z1,x2,y2,z2
    cmd = f"G:{x_pick},{y_pick},{z_pick},{x_place},{y_place},{z_place}\n"
    
    # Send to Arduino
    arm.write(cmd.encode())
    print(f"[SERIAL] Sent sequence to Arduino: {cmd.strip()}")