#include "copy.hpp"

#include "../../number.hpp"

#include <cstring>

namespace rund::node::accel::detail::metalbatch {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

bool pack(const std::span<const BatchMapView> views, const BatchMapPlan &plan,
          const Maps &maps, const Workspace &workspace) {
  auto *const target =
      static_cast<std::byte *>(MetalBufferContents(workspace.input));
  if (plan.input_bytes != 0u && target == nullptr) {
    return false;
  }
  for (std::size_t group_index = 0u; group_index < plan.size; ++group_index) {
    const BatchMapGroup &group = plan.groups[group_index];
    if (!group.packed) {
      continue;
    }
    const BatchMapView &view = views[group.begin];
    for (std::size_t binding = 0u; binding < view.input_count; ++binding) {
      const std::uint64_t bytes = view.tiles * view.inputs[binding];
      for (std::size_t job = 0u; job < group.count; ++job) {
        MetalMapEncodeResources *const map = maps[group.begin + job];
        const auto *const ref =
            map == nullptr ? nullptr
                           : map->bindings.resident_inputs.ref(binding);
        const MetalResidentBufferResult *const resident =
            map == nullptr ? nullptr : &map->resident.input(binding);
        void *const source =
            ref == nullptr || resident == nullptr
                ? nullptr
                : MetalBufferContents(MetalRuntimeBuffer{
                      .bytes = ref->bytes, .buffer = resident->device_buffer});
        const std::uint64_t offset =
            group.inputs[binding] + bytes * static_cast<std::uint64_t>(job);
        std::size_t copy_bytes = 0u;
        std::size_t destination = 0u;
        if (source == nullptr || !ToSize(bytes, copy_bytes) ||
            !ToSize(offset, destination) ||
            destination > workspace.input.bytes ||
            copy_bytes > workspace.input.bytes - destination) {
          return false;
        }
        std::memcpy(target + destination, source, copy_bytes);
      }
    }
  }
  return true;
}

bool unpack(const std::span<const BatchMapView> views, const BatchMapPlan &plan,
            const Maps &maps, const Workspace &workspace) {
  const auto *const source =
      static_cast<const std::byte *>(MetalBufferContents(workspace.output));
  if (plan.output_bytes != 0u && source == nullptr) {
    return false;
  }
  for (std::size_t group_index = 0u; group_index < plan.size; ++group_index) {
    const BatchMapGroup &group = plan.groups[group_index];
    if (!group.packed) {
      continue;
    }
    const BatchMapView &view = views[group.begin];
    for (std::size_t binding = 0u; binding < view.output_count; ++binding) {
      const std::uint64_t bytes = view.tiles * view.outputs[binding];
      for (std::size_t job = 0u; job < group.count; ++job) {
        MetalMapEncodeResources *const map = maps[group.begin + job];
        const auto *const ref =
            map == nullptr ? nullptr
                           : map->bindings.resident_outputs.ref(binding);
        const MetalResidentBufferResult *const resident =
            map == nullptr ? nullptr : &map->resident.output(binding);
        void *const target =
            ref == nullptr || resident == nullptr
                ? nullptr
                : MetalBufferContents(MetalRuntimeBuffer{
                      .bytes = ref->bytes, .buffer = resident->device_buffer});
        const std::uint64_t offset =
            group.outputs[binding] + bytes * static_cast<std::uint64_t>(job);
        std::size_t copy_bytes = 0u;
        std::size_t source_offset = 0u;
        if (target == nullptr || !ToSize(bytes, copy_bytes) ||
            !ToSize(offset, source_offset) ||
            source_offset > workspace.output.bytes ||
            copy_bytes > workspace.output.bytes - source_offset) {
          return false;
        }
        std::memcpy(target, source + source_offset, copy_bytes);
      }
    }
  }
  return true;
}

#endif

} // namespace rund::node::accel::detail::metalbatch
