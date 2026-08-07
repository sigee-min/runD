#pragma once

#include "../../kernel/preparation.hpp"
#include "../../kernel/reset/model.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

struct VulkanResetExecution final {
  std::uint64_t commands{};
  bool shader{};
  bool ok{};
};

// One backend-specific execution-form projection. Standalone dense ranges use
// the transfer fill; strided and Pipeline-private ranges use the reset shader.
// The geometric value remains owned by reset::Range.
[[nodiscard]] constexpr VulkanResetExecution
PlanVulkanResetExecution(const reset::Range range,
                         const KernelPreparationMode mode,
                         const std::uint64_t dispatch_window) noexcept {
  if (!range.valid()) {
    return {};
  }
  const bool shader =
      !range.dense() || IsPipelinePrivatePreparation(mode);
  if (shader && dispatch_window == 0u) {
    return {};
  }
  return VulkanResetExecution{
      .commands = shader ? reset::Commands(range.count(), dispatch_window) : 1u,
      .shader = shader,
      .ok = true,
  };
}

} // namespace rund::node::accel::detail
