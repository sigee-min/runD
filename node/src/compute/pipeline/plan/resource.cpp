#include "resource.hpp"

#include "../../size.hpp"
#include "../../type.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail {

PipelineScheduleResources::PipelineScheduleResources(
    const PipelineBuildState &declared)
    : build(declared),
      internals_(build.internals.size(), PipelineResourceUnassigned) {
  shapes.reserve(std::min(build.binding_count, PipelineResourceCapacity));
  accesses.reserve(build.binding_count);
  external_flags.reserve(
      std::min(build.binding_count, PipelineResourceCapacity));
  publication_accesses.reserve(build.publications.size() * 2u);
  external_.reserve(std::min(build.binding_count, PipelineResourceCapacity));
}

Result<std::uint32_t>
PipelineScheduleResources::admit(const PipelineBinding &binding) {
  const BufferState *external_buffer = nullptr;
  std::uint32_t *internal_ordinal = nullptr;
  std::uint64_t bytes = 0u;
  if (binding.owner == PipelineBinding::external) {
    if (binding.buffer == nullptr) {
      return Result<std::uint32_t>::fail(Reason::BindingInvalid);
    }
    external_buffer = binding.buffer.get();
    const auto found = external_.find(external_buffer);
    if (found != external_.end()) {
      return Result<std::uint32_t>::success(found->second);
    }
    bytes = binding.buffer->bytes;
  } else {
    if (binding.owner >= build.internals.size()) {
      return Result<std::uint32_t>::fail(Reason::PipelineInvalid);
    }
    internal_ordinal = &internals_[binding.owner];
    if (*internal_ordinal != PipelineResourceUnassigned) {
      return Result<std::uint32_t>::success(*internal_ordinal);
    }
    const PipelineInternal &internal = build.internals[binding.owner];
    std::size_t internal_bytes = 0u;
    const std::size_t width = type_bytes(internal.type);
    if (width == 0u || !size::multiply(internal.count, width, internal_bytes)) {
      return Result<std::uint32_t>::fail(Reason::PipelineCapacity);
    }
    bytes = internal_bytes;
  }
  if (shapes.size() >= PipelineResourceCapacity) {
    return Result<std::uint32_t>::fail(Reason::PipelineCapacity);
  }
  const auto ordinal = static_cast<std::uint32_t>(shapes.size());
  shapes.push_back(resource::Resource{
      .id = ordinal + 1u,
      .bytes = bytes,
      .alias_group = ordinal + 1u,
  });
  external_flags.push_back(external_buffer != nullptr ? 1u : 0u);
  if (internal_ordinal != nullptr) {
    *internal_ordinal = ordinal;
  } else {
    external_.emplace(external_buffer, ordinal);
  }
  return Result<std::uint32_t>::success(ordinal);
}

bool PipelineScheduleResources::append(
    std::vector<resource::Access> &destination, const PipelineBinding &binding,
    const std::uint32_t node, const std::uint32_t ordinal,
    const resource::AccessMode mode) {
  if (binding.count == 0u) {
    return true;
  }
  std::size_t offset_bytes = 0u;
  std::size_t stride_bytes = 0u;
  if (binding.element_bytes == 0u ||
      !size::multiply(binding.offset, binding.element_bytes, offset_bytes) ||
      !size::multiply(binding.stride, binding.element_bytes, stride_bytes)) {
    return false;
  }
  destination.push_back(resource::Access{
      .node = node,
      .resource = ordinal + 1u,
      .mode = mode,
      .offset_bytes = offset_bytes,
      .element_bytes = binding.element_bytes,
      .element_count = binding.count,
      .stride_bytes = stride_bytes,
  });
  return true;
}

} // namespace rund::compute::detail
