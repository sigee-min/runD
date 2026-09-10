#include "local.hpp"

#include <kernel/program/compute/metadata.hpp>

#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail::recurrence_source_detail {

bool SourceBindings(
    const rund::kernel::ExecutionMetadata &metadata,
    const std::string_view source, const ComputeApi api,
    const ComputeScalar scalar,
    const std::span<const std::uint64_t> history_pitch_bytes,
    std::array<SourceBinding, RecurrenceBindingCapacity> &inputs,
    std::size_t &input_count,
    std::array<OutputBinding, RecurrenceBindingCapacity> &outputs,
    std::size_t &output_count) noexcept {
  if (!metadata.ok ||
      metadata.binding_accesses.size() != metadata.binding_names.size() ||
      metadata.input_element_bytes.size() != metadata.read_count ||
      metadata.output_element_bytes.size() != metadata.write_count ||
      metadata.binding_accesses.size() > RecurrenceBindingCapacity ||
      metadata.read_count > inputs.size() ||
      metadata.write_count > outputs.size()) {
    return false;
  }
  input_count = 0u;
  output_count = 0u;
  for (std::size_t index = 0u; index < metadata.binding_accesses.size();
       ++index) {
    const rund::kernel::ComputeBindingAccess access =
        metadata.binding_accesses[index];
    if (access == rund::kernel::ComputeBindingAccess::Read) {
      if (input_count >= metadata.input_element_bytes.size() ||
          input_count >= 64u) {
        return false;
      }
      const std::uint64_t bit = std::uint64_t{1u} << input_count;
      const bool direct = (metadata.direct_read_mask & bit) != 0u;
      const bool uniform = (metadata.uniform_read_mask & bit) != 0u;
      if (direct == uniform) {
        return false;
      }
      SourceBinding binding{
          .name = metadata.binding_names[index],
          .element_bytes = metadata.input_element_bytes[input_count],
          .uniform = uniform,
      };
      if (!FindInputLoad(source, api, scalar, binding.name, binding.uniform,
                         binding.load_begin, binding.load_end)) {
        return false;
      }
      inputs[input_count++] = binding;
    } else if (access == rund::kernel::ComputeBindingAccess::Write) {
      if (output_count >= metadata.output_element_bytes.size()) {
        return false;
      }
      OutputBinding binding{
          .name = metadata.binding_names[index],
          .element_bytes = metadata.output_element_bytes[output_count],
          .history_pitch_bytes = history_pitch_bytes.empty()
                                     ? 0u
                                     : history_pitch_bytes[output_count],
      };
      if (!FindOutputStore(source, api, scalar, binding.name,
                           binding.store_begin, binding.value_begin,
                           binding.value_end, binding.store_end)) {
        return false;
      }
      outputs[output_count++] = binding;
    } else {
      return false;
    }
  }
  return input_count == metadata.input_element_bytes.size() &&
         output_count == metadata.output_element_bytes.size();
}

} // namespace rund::node::accel::detail::recurrence_source_detail
