#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../adapter/state.hpp"
#include "../command.hpp"
#include "pipeline.hpp"
#include "execute/run.hpp"

#include <memory>
#include <mutex>
#include <utility>

namespace rund::node::accel::detail {

template <class Desc, class Plan, class Bindings>
rund::AccelCheck
RejectVulkanCollectiveExecute(const rund::AccelDevice &pick, const Desc &desc,
                              const Plan &plan, const Bindings &bindings) {
  (void)pick;
  (void)desc;
  (void)plan;
  (void)bindings;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
}

} // namespace rund::node::accel::detail
