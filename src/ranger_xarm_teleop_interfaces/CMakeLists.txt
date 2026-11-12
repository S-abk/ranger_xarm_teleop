cmake_minimum_required(VERSION 3.8)
project(ranger_xarm_teleop_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArmOwner.msg"
  "msg/HandStatus.msg"
  "msg/InputStatus.msg"
  "msg/TeleopState.msg"
  "srv/SetArmOwner.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
