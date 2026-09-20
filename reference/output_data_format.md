1. 每个动作一个文件夹，命名 data_000 …，50Hz，三件套：

    data/SitChair/
    data_000/
        robot_50hz.npz
        obj_motion_global_50hz.pkl
        contact_labels_50hz.npy
    data_001/
    ...


2. robot_50hz.npz

    字段	        | shape	     | 含义 （关节坐标系/世界坐标系）
    joint_pos      | (T, 29)    | G1 29 DoF 关节角
    joint_vel      | (T, 29)    | 关节速度
    body_pos_w     | (T, 35, 3) | 全身 link 世界坐标位置
    body_quat_w    | (T, 35, 4) | 全身 link 姿态，wxyz
    body_lin_vel_w | (T, 35, 3) | link 线速度
    body_ang_vel_w | (T, 35, 3) | link 角速度
    fps            | (1,)       | 通常 50


3. obj_motion_global_50hz.pkl

    字段	     | shape	  | 含义
    obj_trans   | (T', 3)    | 物体世界位置
    obj_rot     | (T', 3, 3) | 旋转矩阵（加载时转成 quat）
    obj_lin_vel | (T', 3)    | 物体线速度
    obj_ang_vel | (T', 3)    | 物体角速度
    obj_scale   | scalar     | 恒为 1.0

4. contact_labels_50hz.npy

    字段  |	shape	  | 含义
    —    | (T,) bool | 每帧期望接触 (hip_pitch/roll 与 object的接触)


5. 另外 object.obj 
    需要存储为缩放之后的 obj 。 position 和 orientation 也为在 retargeting 中实际采用的位置

6. metadata.json
    joint_pos[:, i] 与 body_*[:, i] 的关节 / link 名称、MuJoCo id、qposadr。
    见 ``joint_pos.index`` 与 ``body_links.index``。
