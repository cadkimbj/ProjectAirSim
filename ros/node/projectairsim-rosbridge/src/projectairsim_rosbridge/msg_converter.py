"""
Copyright (C) Microsoft Corporation. 
Copyright (C) 2025 IAMAI CONSULTING CORP
MIT License.
ROS bridge for Project AirSim: Sensor message conversion module
"""

import math

import geometry_msgs.msg as rosgeommsg
import radar_msgs.msg as rosradarmsg
import sensor_msgs.msg as rossensmsg
import std_msgs.msg as rosstdmsg

import numpy as np

from . import utils
from .node import ROSNode

try:
    from livox_ros_driver2.msg import CustomMsg as LivoxCustomMsg
    from livox_ros_driver2.msg import CustomPoint as LivoxCustomPoint

    LIVOX_CUSTOM_MSG_AVAILABLE = True
except ImportError:
    LivoxCustomMsg = None
    LivoxCustomPoint = None
    LIVOX_CUSTOM_MSG_AVAILABLE = False


class MsgConverter:
    """
    This class contains methods to convert Project AirSim topic messages to and
    from ROS topics messages.
    """

    # -------------------------------------------------------------------------
    # MsgConverter Constants
    # -------------------------------------------------------------------------

    # Global Navigation Satellite System (GNSS) fix type
    GNSS_FIX_NO_FIX = 0
    GNSS_FIX_TIME_ONLY = 1
    GNSS_FIX_2D_FIX = 2
    GNSS_FIX_3D_FIX = 3

    # Initialized covariance matrix-as-array indicating no covariance data
    NO_COVARIANCE_MATRIX = [0.0] * 9

    def __init__(self, ros_node: ROSNode):
        """
        Constructor.

        Arguments:
            ros_node - Project AirSim ROS node object
            robot_base_frame_ids - Mapping from Project AirSim topic name to robot's base transform frame ID
        """
        self.max_depth_mm = 6000  # Maximum depth value from 16UC1 image format, used to convert to [0, 255] monochrome image (millimeters)
        self.ros_node = ros_node  # ROS node
        self.robot_base_frame_ids = (
            {}
        )  # Mapping from Project AirSim topic name to robot's base transform frame ID
        self._last_lidar_stamp_sec_by_topic = {}
        self._last_livox_lidar_stamp_sec_by_topic = {}

    def convert_actual_pose_to_ros(self, projectairsim_topic_name, projectairsim_pose):
        """
        Convert a Project AirSim vehicle pose into a ROS Pose message.

        Arguments:
            projectairsim_topic_name - The Project AirSim topic name
            projectairsim_pose - The vehicle pose received from the Project AirSim topic

        Return:
            (return) - Corresponding ROS Pose message
        """
        posestamped = rosgeommsg.PoseStamped()
        posestamped.header.stamp = self._get_stamp_from_projectairsim_msg(
            projectairsim_pose
        )
        # posestamped.header.frame_id is set by PoseBridgeToROS

        posestamped.pose.position = utils.to_ros_point(projectairsim_pose["position"])
        posestamped.pose.orientation = utils.to_ros_quaternion(
            projectairsim_pose["orientation"]
        )

        return posestamped

    def convert_barometer_to_ros(self, projectairsim_topic_name, projectairsim_msg):
        """
        Convert a Project AirSim barometer sensor message into a ROS FluidPressure message.

        Arguments:
            projectairsim_topic_name - The Project AirSim topic name
            projectairsim_msg - The barometer data received from the Project AirSim topic

        Returns:
            (return) - Corresponding ROS FluidPressure message
        """
        fluid_pressure = rossensmsg.FluidPressure()
        fluid_pressure.header = self._get_standard_ros_header(
            projectairsim_topic_name, projectairsim_msg
        )

        fluid_pressure.fluid_pressure = projectairsim_msg["pressure"]
        fluid_pressure.variance = 0.0

        return fluid_pressure

    def convert_distance_sensor_to_ros(self, projectairsim_topic_name, projectairsim_msg):
        """
        Convert a Project AirSim distance sensor message into a ROS Range message.

        Arguments:
            projectairsim_topic_name - The Project AirSim topic name
            projectairsim_msg - The distance sensor data received from the Project AirSim topic

        Returns:
            (return) - Corresponding ROS Range message
        """
        range_msg = rossensmsg.Range()
        range_msg.header = self._get_standard_ros_header(
            projectairsim_topic_name, projectairsim_msg
        )

        range_msg.radiation_type = rossensmsg.Range.INFRARED
        range_msg.field_of_view = 0.0
        range_msg.min_range = 0.0
        range_msg.max_range = float("inf")
        range_msg.range = projectairsim_msg["current_distance"] * 0.01

        return range_msg

    def convert_desired_pose_from_ros(self, ros_topic_name, ros_posestamped):
        """
        Convert a ROS vehicle pose request into a Project AirSim pose message.

        Arguments:
            topic_sub_handler - Topic handler for this topic
            ros_topic_name - The ROS topic name
            ros_posestamped - The vehicle pose received from the ROS topic

        Return:
            (return) - Corresponding Project AirSim Pose message
        """
        projectairsim_pose = {
            "position": utils.to_projectairsim_position(ros_posestamped.pose.position),
            "orientation": utils.to_projectairsim_quaternion(
                ros_posestamped.pose.orientation
            ),
        }
        return projectairsim_pose

    def convert_gps_to_ros(self, projectairsim_topic_name, projectairsim_msg):
        """
        Convert a Project AirSim GPS sensor message into a ROS NavSatFix message.

        Arguments:
            projectairsim_topic_name - The Project AirSim topic name
            projectairsim_msg - The GPS data received from the Project AirSim topic

        Returns:
            (return) - Corresponding ROS NavSatFix message
        """
        nav_sat_fix = rossensmsg.NavSatFix()
        nav_sat_fix.header = self._get_standard_ros_header(
            projectairsim_topic_name, projectairsim_msg
        )

        nav_sat_fix.status.status = (
            rossensmsg.NavSatStatus.STATUS_SBAS_FIX
            if projectairsim_msg["fix_type"] >= self.GNSS_FIX_2D_FIX
            else rossensmsg.NavSatStatus.STATUS_NO_FIX
        )
        nav_sat_fix.status.service = rossensmsg.NavSatStatus.SERVICE_GPS

        nav_sat_fix.latitude = projectairsim_msg["latitude"]
        nav_sat_fix.longitude = projectairsim_msg["longitude"]
        nav_sat_fix.altitude = projectairsim_msg["altitude"]
        nav_sat_fix.position_covariance = [0.0] * 9
        nav_sat_fix.position_covariance_type = rossensmsg.NavSatFix.COVARIANCE_TYPE_UNKNOWN

        return nav_sat_fix

    def convert_image_to_ros(self, projectairsim_topic_name, projectairsim_image):
        """
        Convert a Project AirSim image message into a ROS image message.

        Arguments:
            projectairsim_topic_name - The Project AirSim topic name
            projectairsim_image - The image message received from the Project AirSim topic

        Return:
            (return) - Corresponding ROS Image message
        """
        if projectairsim_image["encoding"] == "BGR":
            return self.convert_image_bgr8_to_ros(
                projectairsim_topic_name, projectairsim_image
            )
        elif projectairsim_image["encoding"] == "16UC1":
            return self.convert_image_16uc1_to_ros(
                projectairsim_topic_name, projectairsim_image
            )
        else:
            raise ValueError(
                f"Can only handle image encoding BGR or 16UC1, not \"{projectairsim_image['encoding']}\""
            )

    def convert_image_bgr8_to_ros(
        self, projectairsim_topic_name, projectairsim_image_bgr8
    ):
        """
        Convert a Project AirSim bgr8 image message into a ROS image message.

        Arguments:
            projectairsim_topic_name - The Project AirSim topic name
            projectairsim_image - The bgr8 image message received from the Project AirSim topic

        Return:
            (return) - Corresponding ROS Image message
        """
        image = rossensmsg.Image()
        image.header.stamp = self._get_stamp_from_projectairsim_msg(
            projectairsim_image_bgr8
        )
        # image.header.frame_id must be set by caller

        # Get image parameters
        image.height = projectairsim_image_bgr8["height"]
        image.width = projectairsim_image_bgr8["width"]
        image.encoding = "bgr8"
        image.is_bigendian = projectairsim_image_bgr8["big_endian"]

        # Convert image data to uncompressed bitmap data
        image.data = projectairsim_image_bgr8["data"]
        image.step = 3 * image.width

        return image

    def convert_image_16uc1_to_ros(
        self, projectairsim_topic_name, projectairsim_image_16uc1
    ):
        """
        Convert a Project AirSim 16uc1 image message into a ROS image message.

        Arguments:
            projectairsim_topic_name - The Project AirSim topic name
            projectairsim_image - The 16uc1 image message received from the Project AirSim topic

        Return:
            (return) - Corresponding ROS Image message
        """
        image = rossensmsg.Image()
        image.header.stamp = self._get_stamp_from_projectairsim_msg(
            projectairsim_image_16uc1
        )
        # image.header.frame_id must be set by caller

        # Get image parameters
        image.height = projectairsim_image_16uc1["height"]
        image.width = projectairsim_image_16uc1["width"]
        image.encoding = "mono8"
        image.is_bigendian = projectairsim_image_16uc1["big_endian"]

        # Convert image data to uncompressed bitmap data
        nparray = np.fromstring(projectairsim_image_16uc1["data"], dtype="uint16")
        nparray = np.reshape(
            nparray,
            [projectairsim_image_16uc1["height"], projectairsim_image_16uc1["width"]],
        )
        nparray = ((nparray / self.max_depth_mm) * 255).astype("uint8")
        image.data = nparray.tostring()
        image.step = image.width

        return image

    def convert_imu_to_ros(self, projectairsim_topic_name, projectairsim_msg):
        """
        Convert a Project AirSim IMU sensor message into a ROS Imu message.

        Arguments:
            projectairsim_topic_name - The Project AirSim topic name
            projectairsim_msg - The IMU data received from the Project AirSim topic

        Returns:
            (return) - Corresponding ROS Imu message
        """
        imu = rossensmsg.Imu()
        imu.header = self._get_standard_ros_header(
            projectairsim_topic_name, projectairsim_msg
        )

        imu.orientation = utils.to_ros_quaternion(projectairsim_msg["orientation"])
        imu.orientation_covariance = self.NO_COVARIANCE_MATRIX
        imu.angular_velocity = utils.to_ros_position_vector3(
            projectairsim_msg["angular_velocity"]
        )
        imu.angular_velocity_covariance = self.NO_COVARIANCE_MATRIX
        imu.linear_acceleration = utils.to_ros_position_vector3(
            projectairsim_msg["linear_acceleration"]
        )
        imu.linear_acceleration_covariance = self.NO_COVARIANCE_MATRIX

        return imu

    def convert_lidar_to_ros(self, projectairsim_topic_name, projectairsim_lidar):
        """
        Convert a Project AirSim LIDAR point cloud message into a ROS PointCloud2 message.

        Arguments:
            projectairsim_topic_name - The Project AirSim topic name
            projectairsim_lidar - The LIDAR data received from the Project AirSim topic

        Returns:
            (return) - ROS geometry_msg.PointCloud2 message
        """
        point_cloud_airsim = projectairsim_lidar["point_cloud"]
        intensity_cloud = projectairsim_lidar.get("intensity_cloud", [])
        ring_cloud = projectairsim_lidar.get("laser_index_cloud", [])
        num_points = len(point_cloud_airsim) // 3

        timestamp_ns = projectairsim_lidar.get("time_stamp")
        timestamp_sec = timestamp_ns * 1.0e-9 if timestamp_ns is not None else None
        last_timestamp_sec = self._last_lidar_stamp_sec_by_topic.get(
            projectairsim_topic_name
        )
        scan_duration_sec = (
            max(timestamp_sec - last_timestamp_sec, 0.0)
            if timestamp_sec is not None and last_timestamp_sec is not None
            else 0.0
        )
        if timestamp_sec is not None:
            self._last_lidar_stamp_sec_by_topic[projectairsim_topic_name] = (
                timestamp_sec
            )

        # Convert data stream into PointCloud2 records.
        # Convert from Project AirSim's RHS Z-down to ROS's RHS Z-up
        # FAST-LIVO2 lidar_type 2 expects Velodyne-style x/y/z/intensity/time/ring.
        points = [
            (
                point_cloud_airsim[i],
                -point_cloud_airsim[i + 1],
                -point_cloud_airsim[i + 2],
                intensity_cloud[i // 3] if (i // 3) < len(intensity_cloud) else 0.0,
                (
                    scan_duration_sec * 1.0e6 * (i // 3) / (num_points - 1)
                    if num_points > 1
                    else 0.0
                ),
                ring_cloud[i // 3] if (i // 3) < len(ring_cloud) else 0,
            )
            for i in range(0, len(point_cloud_airsim), 3)
        ]

        header = rosstdmsg.Header()
        header.stamp = self._get_stamp_from_projectairsim_msg(projectairsim_lidar)
        header.frame_id = projectairsim_lidar["frame_id"]
        fields = [
            rossensmsg.PointField(
                name="x", offset=0, datatype=rossensmsg.PointField.FLOAT32, count=1
            ),
            rossensmsg.PointField(
                name="y", offset=4, datatype=rossensmsg.PointField.FLOAT32, count=1
            ),
            rossensmsg.PointField(
                name="z", offset=8, datatype=rossensmsg.PointField.FLOAT32, count=1
            ),
            rossensmsg.PointField(
                name="intensity",
                offset=16,
                datatype=rossensmsg.PointField.FLOAT32,
                count=1,
            ),
            rossensmsg.PointField(
                name="time",
                offset=20,
                datatype=rossensmsg.PointField.FLOAT32,
                count=1,
            ),
            rossensmsg.PointField(
                name="ring",
                offset=24,
                datatype=rossensmsg.PointField.UINT16,
                count=1,
            ),
        ]
        pointcloud2 = self.ros_node.PointCloud2.create_cloud(header, fields, points)

        return pointcloud2

    def convert_lidar_to_livox_custom_msg(
        self, projectairsim_topic_name, projectairsim_lidar
    ):
        """
        Convert a Project AirSim LIDAR point cloud message into a Livox CustomMsg.

        This is intended for ROS2 pipelines that consume
        livox_ros_driver2/msg/CustomMsg, such as FAST-LIVO2's Livox input path.
        Project AirSim does not carry per-ray timestamps, so offset_time is
        approximated from each laser ring's point order within the scan.

        Arguments:
            projectairsim_topic_name - The Project AirSim topic name
            projectairsim_lidar - The LIDAR data received from the Project AirSim topic

        Returns:
            (return) - ROS livox_ros_driver2.msg.CustomMsg message
        """
        if not LIVOX_CUSTOM_MSG_AVAILABLE:
            return None

        point_cloud_airsim = projectairsim_lidar["point_cloud"]
        intensity_cloud = projectairsim_lidar.get("intensity_cloud", [])
        ring_cloud = projectairsim_lidar.get("laser_index_cloud", [])
        num_points = len(point_cloud_airsim) // 3

        timestamp_ns = projectairsim_lidar.get("time_stamp")
        scan_duration_sec = self._get_lidar_scan_duration_sec(
            self._last_livox_lidar_stamp_sec_by_topic,
            projectairsim_topic_name,
            timestamp_ns,
        )
        scan_duration_us = max(scan_duration_sec * 1.0e6, 0.0)
        offset_times_us = self._get_livox_offset_times_us(
            ring_cloud, num_points, scan_duration_us
        )

        livox_msg = LivoxCustomMsg()
        livox_msg.header = rosstdmsg.Header()
        livox_msg.header.stamp = self._get_stamp_from_projectairsim_msg(
            projectairsim_lidar
        )
        livox_msg.header.frame_id = projectairsim_lidar["frame_id"]
        livox_msg.timebase = int(timestamp_ns) if timestamp_ns is not None else 0
        livox_msg.lidar_id = 0
        livox_msg.point_num = num_points

        points = []
        for point_index, i in enumerate(range(0, num_points * 3, 3)):
            point = LivoxCustomPoint()
            point.offset_time = self._clamp_int(
                offset_times_us[point_index], 0, 0xFFFFFFFF
            )
            point.x = float(point_cloud_airsim[i])
            point.y = float(-point_cloud_airsim[i + 1])
            point.z = float(-point_cloud_airsim[i + 2])
            point.reflectivity = self._lidar_intensity_to_livox_reflectivity(
                intensity_cloud[point_index]
                if point_index < len(intensity_cloud)
                else 0
            )
            point.tag = 0x10
            point.line = self._clamp_int(
                ring_cloud[point_index] if point_index < len(ring_cloud) else 0,
                0,
                255,
            )
            points.append(point)

        livox_msg.points = points

        return livox_msg

    @staticmethod
    def _clamp_int(value, min_value, max_value):
        return int(max(min_value, min(max_value, round(float(value)))))

    @classmethod
    def _lidar_intensity_to_livox_reflectivity(cls, intensity):
        intensity = max(float(intensity), 0.0)
        return cls._clamp_int((intensity / 1.4) * 255.0, 0, 255)

    @staticmethod
    def _get_livox_offset_times_us(ring_cloud, num_points, scan_duration_us):
        if num_points <= 0:
            return []

        if len(ring_cloud) >= num_points:
            ring_counts = {}
            for point_index in range(num_points):
                ring = int(ring_cloud[point_index])
                ring_counts[ring] = ring_counts.get(ring, 0) + 1

            ring_seen = {}
            offset_times_us = []
            for point_index in range(num_points):
                ring = int(ring_cloud[point_index])
                ring_index = ring_seen.get(ring, 0)
                ring_seen[ring] = ring_index + 1
                ring_count = ring_counts[ring]
                offset_time_us = (
                    scan_duration_us * ring_index / (ring_count - 1)
                    if ring_count > 1
                    else 0.0
                )
                offset_times_us.append(offset_time_us)

            return offset_times_us

        return [
            scan_duration_us * point_index / (num_points - 1)
            if num_points > 1
            else 0.0
            for point_index in range(num_points)
        ]

    @staticmethod
    def _get_lidar_scan_duration_sec(
        last_timestamp_sec_by_topic, projectairsim_topic_name, timestamp_ns
    ):
        timestamp_sec = timestamp_ns * 1.0e-9 if timestamp_ns is not None else None
        last_timestamp_sec = last_timestamp_sec_by_topic.get(projectairsim_topic_name)
        scan_duration_sec = (
            max(timestamp_sec - last_timestamp_sec, 0.0)
            if timestamp_sec is not None and last_timestamp_sec is not None
            else 0.0
        )
        if timestamp_sec is not None:
            last_timestamp_sec_by_topic[projectairsim_topic_name] = timestamp_sec

        return scan_duration_sec

    def convert_lidar_to_ros_transform(
        self, projectairsim_topic_name, projectairsim_msg
    ):
        """
        Returns the ROS transform to the sensor from the parent
        transform frame (usually the vehicle frame.)

        Arguments:
            projectairsim_topic_name - The Project AirSim topic name
            projectairsim_msg - The barometer data received from the Project AirSim topic

        Returns:
            (return) - Corresponding ROS Transform object
        """
        transform = rosgeommsg.Transform()
        projectairsim_pose = projectairsim_msg["pose"]
        (
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ) = utils.to_ros_position_list(projectairsim_pose["position"])
        (
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ) = utils.to_ros_quaternion_list(projectairsim_pose["orientation"])

        return transform

    def convert_magnetometer_to_ros(self, projectairsim_topic_name, projectairsim_msg):
        """
        Convert a Project AirSim magnetometer sensor message into a ROS MagneticField message.

        Arguments:
            projectairsim_topic_name - The Project AirSim topic name
            projectairsim_msg - The magnetometer data received from the Project AirSim topic

        Returns:
            (return) - Corresponding ROS MagneticField message
        """
        magnetic_field = rossensmsg.MagneticField()
        magnetic_field.header = self._get_standard_ros_header(
            projectairsim_topic_name, projectairsim_msg
        )

        magnetic_field.magnetic_field = utils.to_ros_position_vector3(
            projectairsim_msg["magnetic_field_body"]
        )

        magnetic_field.magnetic_field_covariance = self.NO_COVARIANCE_MATRIX
        projectairsim_covariance = projectairsim_msg["magnetic_field_covariance"]
        for i in range(0, min(len(projectairsim_covariance), 9)):
            magnetic_field.magnetic_field_covariance[i] = projectairsim_covariance[i]

        return magnetic_field

    def convert_radar_detection_to_ros(
        self, projectairsim_topic_name, projectairsim_radar_detections
    ):
        """
        Convert a Project AirSim RADAR detections message into a ROS RadarScan message.

        Arguments:
            topic_pub_handler - Topic handler for this topic
            projectairsim_topic_name - The Project AirSim topic name
            projectairsim_radar_detections - The RADAR data received from the Project AirSim topic

        Returns:
            (return) - Corresponding ROS RadarScan message
        """
        radarscan = rosradarmsg.RadarScan()
        radarscan.header = self._get_standard_ros_header(
            projectairsim_topic_name, projectairsim_radar_detections
        )

        radar_returns = radarscan.returns
        rdProjectAirSim = projectairsim_radar_detections["radar_detections"]
        for i in range(len(rdProjectAirSim)):
            radar_detection = rdProjectAirSim[i]
            range_target = radar_detection["range"]

            radar_return = rosradarmsg.RadarReturn()
            radar_return.range = range_target
            radar_return.azimuth = radar_detection["azimuth"]
            radar_return.elevation = radar_detection["elevation"]
            radar_return.doppler_velocity = radar_detection["velocity"]

            # Attempt to convert the radar cross-section to a signal amplitude
            # See: https://en.wikipedia.org/wiki/Radar_cross-section#Measurement
            radar_return.amplitude = 10 * math.log10(
                radar_detection["rcs_sqm"]
                * 1000  # Typical antenna gain (30 dB)
                / (
                    16.0
                    * math.pi
                    * math.pi
                    * range_target
                    * range_target
                    * range_target
                    * range_target
                )  # Spherical spread modeling of power density at transmitter and scattered by target
                * 0.7  # Typical large radar antenna aperture efficiency
                * 15
            )  # Antenna geometric area (meters squared)

            radar_returns.append(radar_return)

        return radarscan

    def convert_radar_track_to_ros(
        self, projectairsim_topic_name, projectairsim_radar_track
    ):
        """
        Converts the Project AirSim radar track data mesage into a ROS RadarTracks message.

        Arguments:
            topic_pub_handler - Topic handler for this topic
            projectairsim_topic_name - The Project AirSim topic name
            projectairsim_radar_track - The RADAR data received from the Project AirSim topic

        Returns:
            (return) - Corresponding ROS RadarTracks message
        """
        radartracks = rosradarmsg.RadarTracks()
        radartracks.header = self._get_standard_ros_header(
            projectairsim_topic_name, projectairsim_radar_track
        )

        tracks = radartracks.tracks
        rdProjectAirSim = projectairsim_radar_track["radar_tracks"]
        for i in range(len(rdProjectAirSim)):
            radar_track = rdProjectAirSim[i]

            radartrack = rosradarmsg.RadarTrack()

            (
                radartrack.position.x,
                radartrack.position.y,
                radartrack.position.z,
            ) = utils.to_ros_position_list(radar_track["position_est"])
            (
                radartrack.velocity.x,
                radartrack.velocity.y,
                radartrack.velocity.z,
            ) = utils.to_ros_position_list(radar_track["velocity_est"])
            (
                radartrack.acceleration.x,
                radartrack.acceleration.y,
                radartrack.acceleration.z,
            ) = utils.to_ros_position_list(radar_track["accel_est"])

            tracks.append(radartrack)

        return radartracks

    def set_robot_base_frame_ids(self, robot_base_frame_ids: list):
        """
        Sets the mapping from Project AirSim topic name to transform frame IDs

        Arguments:
            robot_base_frame_ids - Dictionary mapping topic names to frame IDs
        """
        self.robot_base_frame_ids = robot_base_frame_ids

    def _get_stamp_from_projectairsim_msg(self, projectairsim_msg):
        """
        Returns a ROS timestamp from Project AirSim sim time, with a wall-clock
        fallback for messages that do not provide a simulation timestamp.
        """
        if "time_stamp" in projectairsim_msg:
            return self.ros_node.get_time_msg_from_timestamp_ns(
                projectairsim_msg["time_stamp"]
            )

        return self.ros_node.get_time_now_msg()

    def _get_standard_ros_header(self, projectairsim_topic_name: str, projectairsim_msg):
        """
        Returns a ROS header with the Project AirSim message timestamp and the
        frame ID set to the corresponding robot base's transform frame ID
        """
        header = rosstdmsg.Header()
        header.stamp = self._get_stamp_from_projectairsim_msg(projectairsim_msg)
        header.frame_id = self.robot_base_frame_ids[
            projectairsim_topic_name
        ]

        return header
