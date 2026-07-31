#include <accel/context/buffer.hpp>
#include <accel/graph/visibility.hpp>
#include <accel/kernel/run/binding.hpp>

#include "local.hpp"

#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail {

ProducerConsumerRoundtrip RejectRoundtrip() noexcept {
  return ProducerConsumerRoundtrip{.ok = false,
                                   .reason = "accel_kernel_run_invalid"};
}

bool BindingSpanBytes(const rund::AccelRunBinding &binding,
                      std::uint64_t &bytes) noexcept {
  if (binding.buffer == nullptr || binding.buffer->scalar_width_bytes == 0u) {
    return false;
  }
  const std::uint64_t element = binding.element_bytes == 0u
                                    ? binding.buffer->scalar_width_bytes
                                    : binding.element_bytes;
  const std::uint64_t count = binding.element_count == 0u
                                  ? binding.buffer->count
                                  : binding.element_count;
  return element != 0u && count != 0u &&
         rund::kernel::checked::mul(count, element, bytes);
}

bool SameBinding(const rund::AccelRunBinding &left,
                 const rund::AccelRunBinding &right) noexcept {
  if (left.buffer == nullptr || right.buffer == nullptr ||
      left.buffer->resident.id == 0u ||
      left.buffer->resident.id != right.buffer->resident.id ||
      left.buffer->context_id != right.buffer->context_id) {
    return false;
  }
  const std::uint64_t left_element = left.element_bytes == 0u
                                         ? left.buffer->scalar_width_bytes
                                         : left.element_bytes;
  const std::uint64_t right_element = right.element_bytes == 0u
                                          ? right.buffer->scalar_width_bytes
                                          : right.element_bytes;
  const std::uint64_t left_count =
      left.element_count == 0u ? left.buffer->count : left.element_count;
  const std::uint64_t right_count =
      right.element_count == 0u ? right.buffer->count : right.element_count;
  const std::uint64_t left_stride =
      left.stride_bytes == 0u ? left_element : left.stride_bytes;
  const std::uint64_t right_stride =
      right.stride_bytes == 0u ? right_element : right.stride_bytes;
  return left.offset_bytes == right.offset_bytes &&
         left_element == right_element && left_count == right_count &&
         left_stride == right_stride;
}

bool BindingRoleIs(const KernelExecution &execution,
                   const std::uint64_t binding,
                   const std::uint64_t binding_count,
                   const rund::kernel::BufferRole role) noexcept {
  return binding < binding_count &&
         execution.graph_roles[static_cast<std::size_t>(binding)] == role;
}

bool BindingVisibilityIsInternal(const KernelExecution &execution,
                                 const std::uint64_t binding,
                                 const std::uint64_t binding_count,
                                 bool &internal) noexcept {
  if (binding >= binding_count ||
      binding >= execution.graph_visibilities.size()) {
    return false;
  }
  const rund::GraphBufferVisibility visibility =
      execution.graph_visibilities[static_cast<std::size_t>(binding)];
  if (visibility == rund::GraphBufferVisibility::Internal) {
    internal = true;
    return true;
  }
  if (visibility == rund::GraphBufferVisibility::External) {
    internal = false;
    return true;
  }
  return false;
}

} // namespace rund::node::accel::detail
