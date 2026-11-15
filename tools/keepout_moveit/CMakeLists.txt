# Copyright 2026 Shuaib Olanrewaju
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Stand-alone build of the helper tools/compare_keepout_moveit.py drives (not a colcon package):
#   cmake -S tools/keepout_moveit -B build_dir && cmake --build build_dir
cmake_minimum_required(VERSION 3.8)
project(keepout_moveit CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(moveit_core REQUIRED)
find_package(urdf REQUIRED)
find_package(srdfdom REQUIRED)
add_executable(keepout_moveit keepout_moveit.cpp)
target_link_libraries(keepout_moveit
  moveit_core::moveit_planning_scene
  moveit_core::moveit_collision_detection_fcl
  moveit_core::moveit_robot_model
  moveit_core::moveit_robot_state)
