import struct
import socket
import time
import ctypes
import yaml
import argparse

import general_motion_retargeting.xsense.xsens_shared_memory as xsens_shm


def byte2str(data, n):
    fmt = f"!{n}c"
    char_array = struct.unpack(fmt, data)
    str_out = ""
    for c in char_array:
        s = c.decode("utf-8")
        str_out += s
    return str_out


def byte2float(data):
    return struct.unpack("!f", data)[0]


def byte2uint32(data):
    return struct.unpack("!I", data)[0]


def byte2uint16(data):
    return struct.unpack("!H", data)[0]


def byte2uint8(data):
    return struct.unpack("!B", data)[0]


class Header:
    def __init__(self, header):
        assert isinstance(header, list) and len(header) == 10
        self.id_str = header[0]
        self.sample_counter = header[1]
        self.datagram_counter = header[2]
        self.items_num = header[3]
        self.time_code = header[4]
        self.character_id = header[5]
        self.body_segments_num = header[6]
        self.prop_segments_num = header[7]
        self.finger_segments_num = header[8]
        self.payload_size = header[9]


    def __repr__(self):
        s = "Header: \n" + \
                "\t" + f"ID Sting: {self.id_str}\n" + \
                "\t" + f"Sample Counter: {self.sample_counter}\n" + \
                "\t" + f"Datagram Counter: {self.datagram_counter}\n" + \
                "\t" + f"Time Code: {self.time_code}\n" + \
                "\t" + f"Character ID: {self.character_id}\n" + \
                "\t" + f"Number of Body Segments: {self.body_segments_num}\n" + \
                "\t" + f"Number of Prop Segments: {self.prop_segments_num}\n" + \
                "\t" + f"Number of Finger Tracking Data Segments: {self.finger_segments_num}\n" + \
                "\t" + f"Payload Size: {self.payload_size}"
        return s
    

    @property
    def is_valid(self):
        if self.id_str != "MXTP02":
            print(f"XsensInterface: Currently only support MXTP02, but got {self.id_str}.")
            return False
        if self.payload_size % self.items_num != 0:
            print(f"XsensInterface: Payload with size {self.payload_size} is not dividable by {self.items_num} items.")
            return False
        return True
        
    
    @property
    def item_size(self):
        return self.payload_size // self.items_num
    

class Datagram:
    def __init__(self, header, payload):
        self.header = header
        self.payload = payload

    
    def decode_pose_msg(self):
        pose_msgs = []
        for i in range(self.header.items_num):
            item = self.payload[i * self.header.item_size : (i + 1) * self.header.item_size]
            
            if self.header.id_str == "MXTP02":
                pose_msg = self._decode_pose_msg_segment(item)
        
            if pose_msg is None:
                return None
            
            pose_msgs.append(pose_msg)
        return pose_msgs
    

    @staticmethod
    def _decode_pose_msg_segment(item):
        if len(item) != 32:
            print(f"XsensInterface: Segment data quaternion with size {len(item)} is not equal to 32.")
            return None
        
        segment_id = byte2uint32(item[0:4])
        x = byte2float(item[4:8])
        y = byte2float(item[8:12])
        z = byte2float(item[12:16])
        qw = byte2float(item[16:20])
        qx = byte2float(item[20:24])
        qy = byte2float(item[24:28])
        qz = byte2float(item[28:32])
        segment_pos = [x, y, z]
        segment_quat = [qw, qx, qy, qz]
        return [segment_id, segment_pos, segment_quat]
        
    
    @staticmethod
    def _decode_pose_msg_joint(item):
        if len(item) != 20:
            print(f"XsensInterface: Segment data quaternion with size {len(item)} is not equal to 20.")
            return None
        
        parent_segment_point_id = byte2uint32(item[0:4])
        child_segment_point_id = byte2uint32(item[4:8])

        parent_segment_id = parent_segment_point_id // 256
        child_segment_id = child_segment_point_id // 256
        parent_segment_point_local_id = parent_segment_point_id - parent_segment_id * 256
        child_segment_point_local_id = child_segment_point_id - child_segment_id * 256
        
        rx = byte2float(item[8:12])
        ry = byte2float(item[12:16])
        rz = byte2float(item[16:20])

        segment_id = [parent_segment_id, child_segment_id]
        joint_pos = [rx, ry, rz]
        return [segment_id, joint_pos]
    

class XsensInterface:
    def __init__(self, config):
        self.config = config

        self.udp_port = self.config["xsens"]["udp"]["port"]
        self.buffer_size = self.config["xsens"]["udp"]["buffer_size"]
        self.stream_freq = self.config["xsens"]["udp"]["stream_freq"]

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.bind(("", self.udp_port))

        self.xsens_shm_manager = xsens_shm.XsensShareMemoryManager()


    def get_datagram(self):
        data, client_address = self.server_socket.recvfrom(self.buffer_size)
        header = self._get_header(data)

        if header is not None and header.is_valid:
            datagram = Datagram(header, data[24:])
            pose_msgs = datagram.decode_pose_msg()
            return header, pose_msgs
        else:
            return None, None


    @staticmethod
    def _get_header(data):
        if len(data) < 24:
            print(f"XsensInterface: Header data length {len(data)} is less than 24.")
            return None
        
        id_str = byte2str(data[0:6], 6)
        sample_counter = byte2uint32(data[6:10])
        datagram_counter = struct.unpack("!B", data[10].to_bytes(1, "big"))
        items_num = byte2uint8(data[11:12])
        time_code = byte2uint32(data[12:16])
        character_id = byte2uint8(data[16:17])
        body_segments_num = byte2uint8(data[17:18])
        prop_segments_num = byte2uint8(data[18:19])
        finger_segments_num = byte2uint8(data[19:20])
        payload_size = byte2uint16(data[22:24])
        
        header = Header(
            [
                id_str,
                sample_counter,
                datagram_counter,
                items_num,
                time_code,
                character_id,
                body_segments_num,
                prop_segments_num,
                finger_segments_num,
                payload_size,
            ]
        )
        return header
    

    def run(self):
        while True:
            step_start_time = time.time()

            header, pose_msgs = xsens_interface.get_datagram()

            if isinstance(pose_msgs, list):
                if header.id_str == "MXTP02":
                    self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.id_str = ctypes.c_char_p("MXTP02".encode("utf-8"))
                    self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.items_num = header.items_num
                    self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.time_code = header.time_code
                    self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.character_id = header.character_id
                    self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.body_segments_num = header.body_segments_num
                    self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.prop_segments_num = header.prop_segments_num
                    self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.finger_segments_num = header.finger_segments_num
                    self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.payload_size = header.payload_size
                    
                    for i in range(header.body_segments_num):
                        self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.body_segment[i].id = pose_msgs[i][0]
                        self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.body_segment[i].pos.x = pose_msgs[i][1][0]
                        self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.body_segment[i].pos.y = pose_msgs[i][1][1]
                        self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.body_segment[i].pos.z = pose_msgs[i][1][2]
                        self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.body_segment[i].quat.qw = pose_msgs[i][2][0]
                        self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.body_segment[i].quat.qx = pose_msgs[i][2][1]
                        self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.body_segment[i].quat.qy = pose_msgs[i][2][2]
                        self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.body_segment[i].quat.qz = pose_msgs[i][2][3]

                    if header.prop_segments_num > 0:
                        for i in range(header.prop_segments_num):
                            self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.prop_segment[i].id = pose_msgs[header.body_segments_num + i][0]
                            self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.prop_segment[i].pos.x = pose_msgs[header.body_segments_num + i][1][0]
                            self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.prop_segment[i].pos.y = pose_msgs[header.body_segments_num + i][1][1]
                            self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.prop_segment[i].pos.z = pose_msgs[header.body_segments_num + i][1][2]
                            self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.prop_segment[i].quat.qw = pose_msgs[header.body_segments_num + i][2][0]
                            self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.prop_segment[i].quat.qx = pose_msgs[header.body_segments_num + i][2][1]
                            self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.prop_segment[i].quat.qy = pose_msgs[header.body_segments_num + i][2][2]
                            self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.prop_segment[i].quat.qz = pose_msgs[header.body_segments_num + i][2][3]

                    if header.finger_segments_num > 0:
                        for i in range(header.finger_segments_num):
                            self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.finger_segment[i].id = pose_msgs[header.body_segments_num + header.prop_segments_num + i][0]
                            self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.finger_segment[i].pos.x = pose_msgs[header.body_segments_num + header.prop_segments_num + i][1][0]
                            self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.finger_segment[i].pos.y = pose_msgs[header.body_segments_num + header.prop_segments_num + i][1][1]
                            self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.finger_segment[i].pos.z = pose_msgs[header.body_segments_num + header.prop_segments_num + i][1][2]
                            self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.finger_segment[i].quat.qw = pose_msgs[header.body_segments_num + header.prop_segments_num + i][2][0]
                            self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.finger_segment[i].quat.qx = pose_msgs[header.body_segments_num + header.prop_segments_num + i][2][1]
                            self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.finger_segment[i].quat.qy = pose_msgs[header.body_segments_num + header.prop_segments_num + i][2][2]
                            self.xsens_shm_manager.xsens_data.get_xsens_extracted_data.finger_segment[i].quat.qz = pose_msgs[header.body_segments_num + header.prop_segments_num + i][2][3]

                    self.xsens_shm_manager.copy_in_data()

            step_elapsed_time = time.time() - step_start_time
            step_sleep_time = max(0, 1.0 / self.stream_freq - step_elapsed_time)
            time.sleep(step_sleep_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="xsens.yaml")
    args = parser.parse_args()
    with open(f"./scripts/configs/{args.config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    xsens_interface = XsensInterface(config)
    xsens_interface.run()