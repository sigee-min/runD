#include "model.hpp"

#include "../../../../device/state.hpp"
#include <kernel/core/checked.hpp>

namespace rund::compute::detail {

bool bind_virtual_graph_banks(const GraphProjectionInputs &inputs,
                              GraphProjectionBanks &banks) noexcept {
  if (inputs.prefix->device->backend == Backend::Cpu) {
    for (std::size_t bank = 0u; bank < residency::Pool::BankCount; ++bank) {
      CpuBufferState *const input =
          cpu_buffer(*inputs.input_owner->buffers[bank]);
      CpuBufferState *const intermediate =
          cpu_buffer(*inputs.intermediate_owner->buffers[bank]);
      CpuBufferState *const control = cpu_buffer(*inputs.pool->control[bank]);
      CpuBufferState *const output =
          cpu_buffer(*inputs.output_owner->buffers[bank]);
      if (input == nullptr || intermediate == nullptr || control == nullptr ||
          output == nullptr || input->data == nullptr ||
          intermediate->data == nullptr || control->data == nullptr ||
          output->data == nullptr || input->bytes < inputs.input_arena_bytes ||
          intermediate->bytes < inputs.intermediate_arena_bytes ||
          control->bytes != inputs.control_arena_bytes ||
          output->bytes < inputs.output_arena_bytes) {
        return false;
      }
      banks.input[bank] = input->data.get();
      banks.control[bank] = control->data.get();
      banks.output[bank] = output->data.get();
    }
    return true;
  }

  std::byte *const host = inputs.pool->host_storage.get();
  std::uint64_t bank_bytes = 0u;
  if (host == nullptr ||
      !kernel::checked::add(inputs.host_input_arena_bytes,
                            inputs.host_output_arena_bytes, bank_bytes)) {
    return false;
  }
  for (std::size_t bank = 0u; bank < residency::Pool::BankCount; ++bank) {
    const std::size_t offset = static_cast<std::size_t>(bank * bank_bytes);
    banks.input[bank] = host + offset;
    banks.output[bank] =
        host + offset + static_cast<std::size_t>(inputs.host_input_arena_bytes);
  }
  return true;
}

} // namespace rund::compute::detail
