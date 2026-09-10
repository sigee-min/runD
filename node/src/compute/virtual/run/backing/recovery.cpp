#include "../backing.hpp"

#include "../../backing.hpp"

namespace rund::compute::detail {

Status
validate_virtual_recovery(const VirtualBacking &input,
                          const VirtualBacking &output,
                          const std::uint64_t required_output_bytes) noexcept {
  const std::uint64_t input_recovery =
      VirtualBackingAccess::recovery_bytes(input);
  const std::uint64_t output_recovery =
      VirtualBackingAccess::recovery_bytes(output);
  return input_recovery == 0u && output_recovery <= required_output_bytes
             ? Status::success()
             : Status::fail(Reason::BufferPoisoned);
}

Status
validate_virtual_recovery(const std::span<VirtualBacking *const> inputs,
                          const VirtualBacking &output,
                          const std::uint64_t required_output_bytes) noexcept {
  if (inputs.empty() ||
      VirtualBackingAccess::recovery_bytes(output) > required_output_bytes) {
    return Status::fail(Reason::BufferPoisoned);
  }
  for (const VirtualBacking *const input : inputs) {
    if (input == nullptr ||
        VirtualBackingAccess::recovery_bytes(*input) != 0u) {
      return Status::fail(Reason::BufferPoisoned);
    }
  }
  return Status::success();
}

Status
validate_virtual_run_recovery(const VirtualRunProjection &run,
                              const std::span<VirtualBacking *const> inputs,
                              const VirtualBacking &output) noexcept {
  if (run.graph_execution() || run.poolless_device_vsm()) {
    return validate_virtual_recovery(inputs, output, run.active.output_bytes);
  }
  if (inputs.empty() || inputs.front() == nullptr) {
    return Status::fail(Reason::BufferPoisoned);
  }
  return validate_virtual_recovery(*inputs.front(), output,
                                   run.active.output_bytes);
}

void clear_virtual_recovery(VirtualBacking &backing) noexcept {
  if (VirtualBackingAccess::transaction_provider(backing) != nullptr) {
    return;
  }
  VirtualBackingAccess::clear_recovery(backing);
}

} // namespace rund::compute::detail
