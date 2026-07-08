
import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale tables
        if "human_scale_table_vertical" in ik_config:
            self.human_scale_table_vertical = {
                k: v * ratio for k, v in ik_config["human_scale_table_vertical"].items()
            }
            self.human_scale_table_horizontal = {
                k: v * ratio for k, v in ik_config["human_scale_table_horizontal"].items()
            }
        else:
            # backward compat: single human_scale_table → same scale for both directions
            scaled = {k: v * ratio for k, v in ik_config["human_scale_table"].items()}
            self.human_scale_table_vertical = scaled
            self.human_scale_table_horizontal = scaled
    
        # self.robot_default_pos = np.array(ik_config["default_pos"])
        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])
        # self.collision_pairs = ik_config["collision_pairs"]

        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.frame_to_task1 = {}
        self.frame_to_task2 = {}
        self.frame_to_bodies1 = {}
        self.frame_to_bodies2 = {}
        # backward-compatible flat dicts
        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}
        
        self.limit = []

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 4*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
            
        self.setup_retarget_configuration()
        
        self.ground_offset = 0.0

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)

        self.tasks1 = []
        self.tasks2 = []

        self._build_frame_mappings(
            self.ik_match_table1,
            self.frame_to_task1,
            self.frame_to_bodies1,
            self.tasks1,
            self.task_errors1,
            self.human_body_to_task1,
            self.pos_offsets1,
            self.rot_offsets1,
        )

        self._build_frame_mappings(
            self.ik_match_table2,
            self.frame_to_task2,
            self.frame_to_bodies2,
            self.tasks2,
            self.task_errors2,
            self.human_body_to_task2,
            self.pos_offsets2,
            self.rot_offsets2,
        )

        self.posture_task = mink.PostureTask(self.model, cost=10.0)
        self.tasks2.append(self.posture_task)
        self.tasks1.append(self.posture_task)

        self.damping_task = mink.DampingTask(self.model, cost=1.0)
        self.tasks1.append(self.damping_task)
        self.tasks2.append(self.damping_task)

    def _build_frame_mappings(
        self, ik_match_table, frame_to_task, frame_to_bodies,
        tasks, task_errors, body_to_task, pos_offsets, rot_offsets,
    ):
        for frame_name, entry in ik_match_table.items():
            # Normalize: single entry → list of entries for uniform processing
            if isinstance(entry[0], list):
                entries = entry
            else:
                entries = [entry]

            first = entries[0]
            _, pos_weight, rot_weight, _, _ = first

            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                frame_to_task[frame_name] = task
                tasks.append(task)
                task_errors[task] = []

                body_entries = []
                for body_name, _pw, _rw, pos_offset, rot_offset in entries:
                    po_np = np.array(pos_offset) - self.ground
                    ro_R = R.from_quat(rot_offset, scalar_first=True)
                    body_entries.append((body_name, po_np, ro_R))
                    # backward-compatible flat dicts
                    body_to_task[body_name] = task
                    pos_offsets[body_name] = po_np
                    rot_offsets[body_name] = ro_R
                frame_to_bodies[frame_name] = body_entries
        
        # self.limit.append(mink.ConfigurationLimit(self.model))
        
        # collision_pairs = [
        #     ([
        #     #   "left_hip_pitch_link_collision", "right_hip_pitch_link_collision",
        #     #   "left_knee_pitch_link_collision", "right_knee_pitch_link_collision",
        #     #   "left_ankle_pitch_link_collision", "right_ankle_pitch_link_collision",
        #       "left_ankle_roll_link_collision", "right_ankle_roll_link_collision",
        #     #   "base_link_collision"
        #       ], ["ground"]),
        # ]
        # collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        #     model=self.model,
        #     geom_pairs=collision_pairs,  # type: ignore
        #     minimum_distance_from_collisions=0.005,
        #     collision_detection_distance=0.01,
        # )
        # self.ik_limits.append(collision_avoidance_limit)
        
  
    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name,
                                            self.human_scale_table_vertical,
                                            self.human_scale_table_horizontal)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for frame_name, body_entries in self.frame_to_bodies1.items():
                task = self.frame_to_task1[frame_name]
                self._set_combined_target(task, body_entries, human_data)
            self.posture_task.set_target_from_configuration(self.configuration)

        if self.use_ik_match_table2:
            for frame_name, body_entries in self.frame_to_bodies2.items():
                task = self.frame_to_task2[frame_name]
                self._set_combined_target(task, body_entries, human_data)
            self.posture_task.set_target_from_configuration(self.configuration)
            
            
    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)
                
                next_error = self.error2()
                num_iter += 1
                
            
        return self.configuration.data.qpos.copy()


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, scale_table_vertical, scale_table_horizontal):
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]

        # scale root: vertical (Z) and horizontal (X,Y) separately
        sv = scale_table_vertical[human_root_name]
        sh = scale_table_horizontal[human_root_name]
        scaled_root_pos = root_pos * np.array([sh, sh, sv])

        # scale other body parts in root-relative local frame, per-axis
        for body_name in human_data.keys():
            if body_name not in scale_table_vertical:
                continue
            if body_name == human_root_name:
                continue
            sv_b = scale_table_vertical[body_name]
            sh_b = scale_table_horizontal[body_name]
            local_pos = human_data[body_name][0] - root_pos
            scaled_local = local_pos * np.array([sh_b, sh_b, sv_b])
            human_data_local[body_name] = scaled_local

        # transform back to global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def _set_combined_target(self, task, body_entries, human_data):
        positions = []
        quaternions = []
        for body_name, *_ in body_entries:
            pos, quat = human_data[body_name]
            positions.append(pos)
            quaternions.append(quat)

        if len(body_entries) == 1:
            task.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3(quaternions[0]), positions[0]))
        else:
            avg_pos = np.mean(positions, axis=0)
            avg_quat = self._average_quaternions(quaternions)
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(avg_quat), avg_pos))

    @staticmethod
    def _average_quaternions(quats):
        q = np.array(quats)
        Q = q.T @ q
        _, v = np.linalg.eigh(Q)
        avg = v[:, -1]
        if avg[0] < 0:
            avg = -avg
        return avg

    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.05
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name] = (pos - np.array([0, 0, self.ground_offset]), quat)
        return human_data
