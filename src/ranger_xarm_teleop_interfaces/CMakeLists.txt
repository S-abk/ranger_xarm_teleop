cmake_minimum_required(VERSION 3.8)
project(ranger_xarm_teleop_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArmOwner.msg"
  "msg/EpisodeState.msg"
  "msg/GripperCommand.msg"
  "msg/Haptic.msg"
  "msg/HandStatus.msg"
  "msg/InputStatus.msg"
  "msg/OperatorStatus.msg"
  "msg/ResetHeartbeat.msg"
  "msg/ResetState.msg"
  "msg/ResetWatchdog.msg"
  "msg/TeleopState.msg"
  "srv/GetJointPath.srv"
  "srv/SetArmOwner.srv"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs trajectory_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
