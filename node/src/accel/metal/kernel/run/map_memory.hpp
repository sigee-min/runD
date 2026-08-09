#pragma once

#include "../../../kernel/backend/template_plan.hpp"
#include "../../runtime/map/resources.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] inline bool
AddMetalHostBytes(std::uint64_t &target, const std::uint64_t count,
                  const std::uint64_t element) noexcept {
  std::uint64_t bytes = 0u;
  return backend_template_plan::product(count, element, bytes) &&
         backend_template_plan::add(target, bytes);
}

[[nodiscard]] inline bool
AddMetalMapTemplateHostBytes(std::uint64_t &target,
                             const rund::kernel::ComputePlan &plan,
                             const std::uint64_t check_count) noexcept {
  return backend_template_plan::add(target,
                                    sizeof(MetalMapTemplateResources)) &&
         AddMetalHostBytes(target, plan.input_buffer_count,
                           sizeof(InputWindowPlan)) &&
         AddMetalHostBytes(target, plan.input_buffer_count,
                           sizeof(std::uint64_t)) &&
         AddMetalHostBytes(target, plan.output_buffer_count,
                           sizeof(std::uint64_t)) &&
         AddMetalHostBytes(target, check_count, sizeof(MetalMapCheck));
}

[[nodiscard]] inline bool
AddMetalMapRouteHostBytes(std::uint64_t &target,
                          const rund::kernel::ComputePlan &plan) noexcept {
  return backend_template_plan::add(target, sizeof(MetalMapEncodeResources)) &&
         AddMetalHostBytes(target,
                           plan.input_buffer_count > kInlineMetalBufferCount
                               ? plan.input_buffer_count -
                                     kInlineMetalBufferCount
                               : 0u,
                           sizeof(MetalResidentBufferResult)) &&
         AddMetalHostBytes(target,
                           plan.output_buffer_count > kInlineMetalBufferCount
                               ? plan.output_buffer_count -
                                     kInlineMetalBufferCount
                               : 0u,
                           sizeof(MetalResidentBufferResult));
}

#endif

} // namespace rund::node::accel::detail
