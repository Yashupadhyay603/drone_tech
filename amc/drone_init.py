import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt


from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool


class start_drone():

    def __init__(self):
        parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
        parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
        parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
        parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
        parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
        parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
        parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
        parser.add_argument('--plot',               default=False,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
        parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
        parser.add_argument('--aggregate',          default=False,      type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
        parser.add_argument('--obstacles',          default=False,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
        parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
        parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
        parser.add_argument('--duration_sec',       default=1000,          type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')

        self.ARGS = parser.parse_args()

        H = .1
        H_STEP = .05
        R = .3
        INIT_XYZS = np.array([[R * np.cos((i / 6) * 2 * np.pi + np.pi / 2),
                               R * np.sin((i / 6) * 2 * np.pi + np.pi / 2) - R, H + i * H_STEP] for i in
                              range(self.ARGS.num_drones)])
        INIT_RPYS = np.array([[0, 0, i * (np.pi / 2) / self.ARGS.num_drones] for i in range(self.ARGS.num_drones)])
        AGGR_PHY_STEPS = int(self.ARGS.simulation_freq_hz / self.ARGS.control_freq_hz) if self.ARGS.aggregate else 1

        if self.ARGS.vision:
            self.env = VisionAviary(drone_model=self.ARGS.drone,
                               num_drones=self.ARGS.num_drones,
                               initial_xyzs=INIT_XYZS,
                               initial_rpys=INIT_RPYS,
                               physics=self.ARGS.physics,
                               neighbourhood_radius=10,
                               freq=self.ARGS.simulation_freq_hz,
                               aggregate_phy_steps=AGGR_PHY_STEPS,
                               gui=self.ARGS.gui,
                               record=self.ARGS.record_video,
                               obstacles=self.ARGS.obstacles
                               )
        else:
            self.env = CtrlAviary(drone_model=self.ARGS.drone,
                             num_drones=self.ARGS.num_drones,
                             initial_xyzs=INIT_XYZS,
                             initial_rpys=INIT_RPYS,
                             physics=self.ARGS.physics,
                             neighbourhood_radius=10,
                             freq=self.ARGS.simulation_freq_hz,
                             aggregate_phy_steps=AGGR_PHY_STEPS,
                             gui=self.ARGS.gui,
                             record=self.ARGS.record_video,
                             obstacles=self.ARGS.obstacles,
                             user_debug_gui=self.ARGS.user_debug_gui
                             )

        PYB_CLIENT = self.env.getPyBulletClient()

        if self.ARGS.drone in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = [DSLPIDControl(self.env) for i in range(self.ARGS.num_drones)]
        elif self.ARGS.drone in [DroneModel.HB]:
            self.ctrl = [SimplePIDControl(self.env) for i in range(self.ARGS.num_drones)]

        self.CTRL_EVERY_N_STEPS = int(np.floor(self.env.SIM_FREQ / self.ARGS.control_freq_hz))
        self.action = {str(i): np.array([0, 0, 0, 0]) for i in range(self.ARGS.num_drones)}
        self.START = time.time()

    def move(self,iterator,target_pos,target_ori):
        obs, reward, done, info = self.env.step(self.action)
        #### Compute control at the desired frequency ##############
        if iterator % self.CTRL_EVERY_N_STEPS == 0:
            #### Compute control for the current way point #############
            for j in range(self.ARGS.num_drones):
                self.action[str(j)], _, _ = self.ctrl[j].computeControlFromState(
                    state=obs[str(j)]["state"],
                    target_pos=np.array(target_pos),
                    target_rpy=np.array(target_ori)
                )
        #### Sync the simulation ###################################
        if self.ARGS.gui:
            sync(iterator, self.START, self.env.TIMESTEP)


