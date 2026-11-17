// Copyright 2026 Shuaib Olanrewaju
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// MoveIt's own collision checker, in process, driven line by line from stdin.
//
//   keepout_moveit URDF_FILE SRDF_FILE
//
// Why this exists: the /check_state_validity service checks robot against robot without
// padding (CollisionRequest::pad_self_collisions is false and the service cannot change it),
// so padding the planning scene has no effect on it. Here the same collision environment move_group
// uses (planning_scene::PlanningScene, FCL, the SRDF's allowed-collision matrix) is asked
// with the padded self-collision check.
//
// Commands (one per line, replies one line each):
//   joints N1 N2 ...        joint names that the following postures give values for
//   pad P                   set padding P (m) on every link with collision geometry
//   acm                     the allowed (never checked) pairs:  acm A|B A|B ...
//   contacts MODE V1 ...    contacts at posture V (other joints 0), through
//                           PlanningScene::checkSelfCollision with pad_self_collisions true
//                           (MODE padded) or false (plain: what the service does), or straight
//                           from the padded (env_padded) or unpadded (unpadded) collision
//                           environment.   c N A|B A|B ...
//   cp P V1 ...             pad P, then contacts padded
//   dist V1 ...             FCL distance of every checked pair (unpadded):  d A|B X ...
//   links                   links with collision geometry:  links L1 L2 ...
//   quit

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/collision_detection/collision_env.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <srdfdom/model.h>
#include <urdf/model.h>

namespace
{
std::string slurp(const char* path)
{
  std::ifstream in(path);
  if (!in)
  {
    std::fprintf(stderr, "cannot read %s\n", path);
    std::exit(2);
  }
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string pair_name(const std::string& a, const std::string& b)
{
  return a < b ? a + "|" + b : b + "|" + a;
}
}  // namespace

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::fprintf(stderr, "usage: keepout_moveit URDF SRDF\n");
    return 2;
  }
  auto urdf_model = std::make_shared<urdf::Model>();
  if (!urdf_model->initString(slurp(argv[1])))
  {
    std::fprintf(stderr, "bad URDF\n");
    return 2;
  }
  auto srdf_model = std::make_shared<srdf::Model>();
  if (!srdf_model->initString(*urdf_model, slurp(argv[2])))
  {
    std::fprintf(stderr, "bad SRDF\n");
    return 2;
  }
  auto model = std::make_shared<moveit::core::RobotModel>(urdf_model, srdf_model);
  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  std::vector<std::string> links;
  for (const auto* link : model->getLinkModelsWithCollisionGeometry())
    links.push_back(link->getName());
  moveit::core::RobotState state(model);
  std::vector<std::string> joints;
  double padding = -1.0;

  auto set_padding = [&](double p) {
    if (p == padding)
      return;
    std::map<std::string, double> map;
    for (const auto& name : links)
      map[name] = p;
    scene->getCollisionEnvNonConst()->setLinkPadding(map);
    padding = p;
  };
  set_padding(0.0);

  auto read_posture = [&](std::istringstream& in) {
    state.setToDefaultValues();
    for (const auto& name : joints)
    {
      double value;
      in >> value;
      state.setVariablePosition(name, value);
    }
    state.update();
  };

  auto contacts = [&](const std::string& mode) {
    collision_detection::CollisionRequest req;
    collision_detection::CollisionResult res;
    req.contacts = true;
    req.max_contacts = 1000;
    req.max_contacts_per_pair = 1;
    req.pad_self_collisions = (mode == "padded");
    if (mode == "env_padded")
      scene->getCollisionEnv()->checkSelfCollision(req, res, state, scene->getAllowedCollisionMatrix());
    else if (mode == "unpadded")
      scene->getCollisionEnvUnpadded()->checkSelfCollision(req, res, state,
                                                           scene->getAllowedCollisionMatrix());
    else
      scene->checkSelfCollision(req, res, state);   // what /check_state_validity calls
    std::set<std::string> names;
    for (const auto& kv : res.contacts)
      names.insert(pair_name(kv.first.first, kv.first.second));
    std::printf("c %zu", names.size());
    for (const auto& n : names)
      std::printf(" %s", n.c_str());
    std::printf("\n");
  };

  std::printf("ready\n");
  std::fflush(stdout);
  std::string line;
  while (std::getline(std::cin, line))
  {
    std::istringstream in(line);
    std::string cmd;
    in >> cmd;
    if (cmd == "quit")
      break;
    if (cmd == "joints")
    {
      joints.assign(std::istream_iterator<std::string>(in), std::istream_iterator<std::string>());
      std::printf("ok %zu\n", joints.size());
    }
    else if (cmd == "pad")
    {
      double p;
      in >> p;
      set_padding(p);
      std::printf("ok\n");
    }
    else if (cmd == "links")
    {
      std::printf("links");
      for (const auto& n : links)
        std::printf(" %s", n.c_str());
      std::printf("\n");
    }
    else if (cmd == "acm")
    {
      const auto& acm = scene->getAllowedCollisionMatrix();
      std::vector<std::string> names;
      acm.getAllEntryNames(names);
      std::set<std::string> out;
      for (const auto& a : names)
        for (const auto& b : names)
        {
          collision_detection::AllowedCollision::Type type;
          if (a < b && acm.getEntry(a, b, type) && type == collision_detection::AllowedCollision::ALWAYS)
            out.insert(pair_name(a, b));
        }
      std::printf("acm");
      for (const auto& n : out)
        std::printf(" %s", n.c_str());
      std::printf("\n");
    }
    else if (cmd == "contacts")
    {
      std::string mode;
      in >> mode;
      read_posture(in);
      contacts(mode);
    }
    else if (cmd == "cp")
    {
      double p;
      in >> p;
      set_padding(p);
      read_posture(in);
      contacts("padded");
    }
    else if (cmd == "dist")
    {
      read_posture(in);
      collision_detection::DistanceRequest req;
      collision_detection::DistanceResult res;
      req.enable_signed_distance = true;
      req.type = collision_detection::DistanceRequestType::ALL;
      req.acm = &scene->getAllowedCollisionMatrix();
      req.distance_threshold = 0.2;
      req.max_contacts_per_body = 1;
      scene->getCollisionEnvUnpadded()->distanceSelf(req, res, state);
      std::map<std::string, double> best;
      for (const auto& kv : res.distances)
        for (const auto& d : kv.second)
        {
          const std::string n = pair_name(kv.first.first, kv.first.second);
          auto it = best.find(n);
          if (it == best.end() || d.distance < it->second)
            best[n] = d.distance;
        }
      std::printf("d");
      for (const auto& kv : best)
        std::printf(" %s %.6f", kv.first.c_str(), kv.second);
      std::printf("\n");
    }
    else
    {
      std::printf("error unknown command: %s\n", cmd.c_str());
    }
    std::fflush(stdout);
  }
  return 0;
}
