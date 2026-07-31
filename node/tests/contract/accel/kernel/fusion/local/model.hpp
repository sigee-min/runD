#pragma once

#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/visibility.hpp>
#include <accel/kernel/run/binding.hpp>

#include <accel/graph/factory/map.hpp>
#include <kernel/program/compute/dsl.hpp>

#include <array>

namespace node_accel_contract::fusion {

using GraphBufferRef = rund::AccelGraphBufferRef;
using GraphNode = rund::AccelGraphNode;
using KernelBinding = rund::AccelRunBinding;
using Role = rund::kernel::BufferRole;
using Visibility = rund::GraphBufferVisibility;

struct Inputs {
  std::array<rund::kernel::i32, 8u> host{};
  std::array<rund::kernel::i32, 8u> vel{};
  std::array<rund::kernel::i32, 8u> add14{};
  std::array<rund::kernel::i32, 8u> add21{};
  std::array<rund::kernel::i32, 8u> add7_vel{};
  std::array<rund::kernel::i32, 8u> add_vel{};
};

} // namespace node_accel_contract::fusion
