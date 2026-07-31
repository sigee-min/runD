#pragma once

#include "../../buffer/resident/model.hpp"

#include <cstddef>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
struct VulkanResidentBindings {
  const rund::kernel::BindingSet *bindings = nullptr;
  std::vector<VulkanResidentBufferResult> inputs{};
  std::vector<VulkanResidentBufferResult> outputs{};
  const char *reason = "ok";

  [[nodiscard]] const VulkanResidentBufferResult &
  input(const rund::kernel::u64 index) const {
    return inputs[static_cast<std::size_t>(index)];
  }

  [[nodiscard]] const VulkanResidentBufferResult &
  output(const rund::kernel::u64 index) const {
    return outputs[static_cast<std::size_t>(index)];
  }
};
#endif

} // namespace rund::node::accel::detail
