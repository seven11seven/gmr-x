import time
import general_motion_retargeting.xsense.xsens_shared_memory as xsens_shm
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
import threading
import argparse
import numpy as np

segment_id_map = {
    0 : "pelvis",
    1 : "L5",
    2 : "L3",
    3 : "T12",
    4 : "T8",
    5 : "Neck",
    6 : "Head",
    7 : "RightShoulder",
    8 : "RightUpperArm",
    9 : "RightForearm",
    10 : "RightHand",
    11 : "LeftShoulder",
    12 : "LeftUpperArm",
    13 : "LeftForearm",
    14 : "LeftHand",
    15 : "RightUpperLeg",
    16 : "RightLowerLeg",
    17 : "RightFoot",
    18 : "RightToe",
    19 : "LeftUpperLeg",
    20 : "LeftLowerLeg",
    21 : "LeftFoot",
    22 : "LeftToe"
}

human_data = {}

for segment_id in segment_id_map.keys():
    human_data[segment_id_map[segment_id]] = [np.zeros(3), np.zeros(4)]
human_data["LeftFootMod"] = [np.zeros(3), np.zeros(4)]
human_data["RightFootMod"] = [np.zeros(3), np.zeros(4)]


def run(shm_manager, args):
    step_start_time = 0.0
    while True:
        step_start_time = time.time()
        shm_manager.copy_out_data()
        print("time_code: ", shm_manager.xsens_data.get_xsens_extracted_data.time_code)
        for sgement_id in segment_id_map.keys():
            
            # print("\tid: ", shm_manager.xsens_data.get_xsens_extracted_data.body_segment[sgement_id - 1].id)
            t_pos = np.array([shm_manager.xsens_data.get_xsens_extracted_data.body_segment[sgement_id ].pos.x,
                              shm_manager.xsens_data.get_xsens_extracted_data.body_segment[sgement_id].pos.y,
                              shm_manager.xsens_data.get_xsens_extracted_data.body_segment[sgement_id].pos.z])
            # print("\tpos: ", t_pos)
            t_quat = np.array([shm_manager.xsens_data.get_xsens_extracted_data.body_segment[sgement_id].quat.qw,
                              shm_manager.xsens_data.get_xsens_extracted_data.body_segment[sgement_id].quat.qx,
                              shm_manager.xsens_data.get_xsens_extracted_data.body_segment[sgement_id].quat.qy,
                              shm_manager.xsens_data.get_xsens_extracted_data.body_segment[sgement_id].quat.qz])
            # print("\tquat: ", t_quat)
            
            human_data[segment_id_map[sgement_id]] = [t_pos, t_quat]
        human_data["LeftFootMod"] = [human_data["LeftFoot"][0], human_data["LeftToe"][1]]
        human_data["RightFootMod"] = [human_data["RightFoot"][0], human_data["RightToe"][1]]

        finger_segments_num = shm_manager.xsens_data.get_xsens_extracted_data.finger_segments_num
        # print("finger segments num: ", shm_manager.xsens_data.get_xsens_extracted_data.finger_segments_num)
        if finger_segments_num > 0:
            print("finger segment: ")
            for i in range(finger_segments_num):
                print("\tid: ", shm_manager.xsens_data.get_xsens_extracted_data.finger_segment[i].id)
                print("\tpos: ")
                print("\t\tx: ", shm_manager.xsens_data.get_xsens_extracted_data.finger_segment[i].pos.x)
                print("\t\ty: ", shm_manager.xsens_data.get_xsens_extracted_data.finger_segment[i].pos.y)
                print("\t\tz: ", shm_manager.xsens_data.get_xsens_extracted_data.finger_segment[i].pos.z)
                print("\tquat: ")
                print("\t\tqw: ", shm_manager.xsens_data.get_xsens_extracted_data.finger_segment[i].quat.qw)
                print("\t\tqx: ", shm_manager.xsens_data.get_xsens_extracted_data.finger_segment[i].quat.qx)
                print("\t\tqy: ", shm_manager.xsens_data.get_xsens_extracted_data.finger_segment[i].quat.qy)
                print("\t\tqz: ", shm_manager.xsens_data.get_xsens_extracted_data.finger_segment[i].quat.qz, "\n")

        step_elapsed_time = time.time() - step_start_time
        step_sleep_time = max(0, 1.0 / args.freq - step_elapsed_time)
        time.sleep(step_sleep_time)
    

def main(args):
    
    shm_manager = xsens_shm.XsensShareMemoryManager()

    # start a thread to client.run()
    thread = threading.Thread(target=run, args=(shm_manager,args))
    thread.start()
    
    print("Starting motion retargeting...")

    retarget = GMR(
            src_human="xsens_realtime",
            tgt_robot=args.robot,
            actual_human_height=1.7,
        )
    viewer = RobotMotionViewer(robot_type=args.robot)

    while True:
        qpos = retarget.retarget(human_data)
        viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            rate_limit=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", type=int, default=120)
    parser.add_argument("--robot", type=str, default="unitree_g1")
    args = parser.parse_args()
    main(args)
    