import ctypes
import sysv_ipc
import atexit


BODY_SEGMENTS_NUM = 23          # fixed
MAX_PROP_SEGMENT_NUM = 4
MAX_FINGER_SEGMENTS_NUM = 40
JOINT_NUM = 28                  # fixed


class SEGMENT_POS(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float), # cm
        ("y", ctypes.c_float),
        ("z", ctypes.c_float)
    ]


class SEGMENT_QUAT(ctypes.Structure):
    _fields_ = [
        ("qw", ctypes.c_float),
        ("qx", ctypes.c_float),
        ("qy", ctypes.c_float),
        ("qz", ctypes.c_float)
    ]


class SEGMENT(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint),
        ("pos", SEGMENT_POS),
        ("quat", SEGMENT_QUAT),
    ]


class JOINT_POS(ctypes.Structure):
    _fields_ = [
        ("rx", ctypes.c_float), # deg
        ("ry", ctypes.c_float),
        ("rz", ctypes.c_float)
    ]


class JOINT(ctypes.Structure):
    _fields_ = [
        ("parent_segment_id", ctypes.c_uint),
        ("child_segment_id", ctypes.c_uint),
        ("pos", JOINT_POS)
    ]


class GET_XSENS_EXTRACTED_DATA(ctypes.Structure):
    _fields_ = [
        ("get_new_data", ctypes.c_int),
        ("id_str", ctypes.c_char_p),
        ("items_num", ctypes.c_uint8),
        ("time_code", ctypes.c_uint),
        ("character_id", ctypes.c_uint8),
        ("body_segments_num", ctypes.c_uint8),
        ("prop_segments_num", ctypes.c_uint8),
        ("finger_segments_num", ctypes.c_uint8),
        ("payload_size", ctypes.c_uint16),
        ("body_segment", SEGMENT * BODY_SEGMENTS_NUM),
        ("prop_segment", SEGMENT * MAX_PROP_SEGMENT_NUM),
        ("finger_segment", SEGMENT * MAX_FINGER_SEGMENTS_NUM),
        ("joint", JOINT * JOINT_NUM)
    ]


class pthread_mutex_t(ctypes.Structure):
        _fields_ = [
        ("sm_mutex", ctypes.c_char * 50)  # range: [0, 1000]
    ]


class XsensShareMemory(ctypes.Structure):
    _fields_ = [
        ("get_xsens_extracted_data", GET_XSENS_EXTRACTED_DATA),
        ("sm_mutex", pthread_mutex_t)
    ]


# Manager class to handle shared memory and mutex
class XsensShareMemoryManager:
    def __init__(self):
        self.key = sysv_ipc.ftok("/tmp/xsenskey", ord("A"))
        self.size = ctypes.sizeof(XsensShareMemory)
        # Create or attach to the shared memory segment
        try:
            self.shm = sysv_ipc.SharedMemory(
                self.key,
                sysv_ipc.IPC_CREAT,
                mode=0o666,
                size=self.size
            )
            print(f"Shared memory segment created with key {self.key}.")
        except sysv_ipc.ExistentialError:
            # If it already exists, just attach to it
            self.shm = sysv_ipc.SharedMemory(self.key)
            print(f"Attached to existing shared memory segment with key {self.key}.")
        
        # Attach to the shared memory segment
        self.shm.attach()
        
        # Get the address of the shared memory
        address = self.shm.address

        if address == 0 or address is None:
            raise RuntimeError("Failed to attach shared memory segment.")
        
        # Map the shared memory to the BipedShareMemory structure
        self.shared_data = XsensShareMemory.from_address(address)
        
        # Create a semaphore for mutual exclusion
        mutex_key = self.key + 1  # Use a different key for the semaphore
        
        try:
            self.mutex = sysv_ipc.Semaphore(
                mutex_key,
                sysv_ipc.IPC_CREAT,
                initial_value=1,
                mode=0o666
            )
            print(f"Semaphore created with key {mutex_key}.")
        except sysv_ipc.ExistentialError:
            # If it already exists, just attach to it
            self.mutex = sysv_ipc.Semaphore(mutex_key)
            print(f"Attached to existing semaphore with key {mutex_key}.")
        
        # Initialize local copies of the data structures
        self.xsens_data = XsensShareMemory()
        
        # Register cleanup to run at exit 
        # no root privilege here
        # atexit.register(self.cleanup)

    
    def cleanup(self):
        """
        Detach and remove shared memory and semaphore.
        """
        if self.shm:
            try:
                self.shm.detach()
                self.shm.remove()
                print("Shared memory detached and removed successfully.")
            except sysv_ipc.ExistentialError:
                print("Shared memory was already removed or not found.")
        if self.mutex:
            try:
                self.mutex.remove()
                print("Semaphore removed successfully.")
            except sysv_ipc.ExistentialError:
                print("Semaphore was already removed or not found.")
    
    
    def lock_memory(self):
        self.mutex.acquire()

    
    def unlock_memory(self):
        self.mutex.release()

    
    def copy_out_data(self):
        """
        Copy data from shared memory to local buffer.
        """
        self.lock_memory()
        try:
            ctypes.memmove(
                ctypes.byref(self.xsens_data.get_xsens_extracted_data),
                ctypes.byref(self.shared_data.get_xsens_extracted_data),
                ctypes.sizeof(GET_XSENS_EXTRACTED_DATA)
            )
        finally:
            self.unlock_memory()


    def copy_in_data(self):
        """
        Copy data from local buffer to shared memory.
        """
        self.lock_memory()
        try:
            ctypes.memmove(
                ctypes.byref(self.shared_data.get_xsens_extracted_data),
                ctypes.byref(self.xsens_data.get_xsens_extracted_data),
                ctypes.sizeof(GET_XSENS_EXTRACTED_DATA)
            )
        finally:
            self.unlock_memory()