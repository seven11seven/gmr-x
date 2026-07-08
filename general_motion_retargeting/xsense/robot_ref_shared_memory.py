import ctypes
import sysv_ipc

# 和 xsens_shared_memory.py 一样的 mutex 结构
class pthread_mutex_t(ctypes.Structure):
    _fields_ = [
        ("sm_mutex", ctypes.c_char * 50)
    ]


class ROBOT_REF_DATA(ctypes.Structure):
    _fields_ = [
        ("frame_id", ctypes.c_uint32),
        ("timestamp", ctypes.c_double),         # 参考帧时间戳（monotonic seconds）
        ("root_pos", ctypes.c_float * 3),      # 机器人根位置 (x,y,z)
        ("root_quat", ctypes.c_float * 4),     # 机器人根姿态 wxyz
        ("joint_pos", ctypes.c_float * 29),    # 29 DOF 关节角（和 num_actions 对应）
    ]


class RobotRefShareMemory(ctypes.Structure):
    _fields_ = [
        ("data", ROBOT_REF_DATA),
        ("sm_mutex", pthread_mutex_t),
    ]


class RobotRefShareMemoryManager:
    """
    仿照 XsensShareMemoryManager，提供 copy_in_data / copy_out_data。
    """

    def __init__(self):
        # 使用不同的 key 文件和字符，避免和 Xsens 冲突
        self.key = sysv_ipc.ftok("/tmp/gmrrefkey", ord("B"))
        self.size = ctypes.sizeof(RobotRefShareMemory)

        try:
            self.shm = sysv_ipc.SharedMemory(
                self.key,
                sysv_ipc.IPC_CREAT,
                mode=0o666,
                size=self.size,
            )
            print(f"[RobotRefSHM] Shared memory segment created with key {self.key}.")
        except sysv_ipc.ExistentialError:
            self.shm = sysv_ipc.SharedMemory(self.key)
            print(f"[RobotRefSHM] Attached to existing shared memory segment with key {self.key}.")

        self.shm.attach()
        address = self.shm.address
        if address == 0 or address is None:
            raise RuntimeError("Failed to attach RobotRef shared memory segment.")

        self.shared_data = RobotRefShareMemory.from_address(address)

        # semaphore
        mutex_key = self.key + 1
        try:
            self.mutex = sysv_ipc.Semaphore(
                mutex_key,
                sysv_ipc.IPC_CREAT,
                initial_value=1,
                mode=0o666,
            )
            print(f"[RobotRefSHM] Semaphore created with key {mutex_key}.")
        except sysv_ipc.ExistentialError:
            self.mutex = sysv_ipc.Semaphore(mutex_key)
            print(f"[RobotRefSHM] Attached to existing semaphore with key {mutex_key}.")

        # 本地缓存
        self.local_data = RobotRefShareMemory()

    def lock_memory(self):
        self.mutex.acquire()

    def unlock_memory(self):
        self.mutex.release()

    def copy_in_data(self):
        """
        Producer：将 self.local_data 写入共享内存
        """
        self.lock_memory()
        try:
            ctypes.memmove(
                ctypes.byref(self.shared_data.data),
                ctypes.byref(self.local_data.data),
                ctypes.sizeof(ROBOT_REF_DATA),
            )
        finally:
            self.unlock_memory()

    def copy_out_data(self):
        """
        Consumer：从共享内存读到 self.local_data
        """
        self.lock_memory()
        try:
            ctypes.memmove(
                ctypes.byref(self.local_data.data),
                ctypes.byref(self.shared_data.data),
                ctypes.sizeof(ROBOT_REF_DATA),
            )
        finally:
            self.unlock_memory()