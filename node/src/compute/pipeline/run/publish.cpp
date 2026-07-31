#include "../local.hpp"
#include "../state.hpp"

#include "../../cpu/view.hpp"

#include <cstddef>
#include <cstring>

namespace rund::compute::detail {

Status publish_cpu_pipeline(PipelineState &state) noexcept {
  if (state.device == nullptr || state.device->backend != Backend::Cpu) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (const PipelinePublish &publication : state.publications) {
    if (publication.source == publication.target || publication.window == 0u ||
        publication.window > state.windows.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    CpuView source{};
    const Status selected =
        cpu_resident_view(state, state.windows[publication.window - 1u],
                          publication.output, source);
    const std::optional<CpuView> target = cpu_view(
        publication.target.get(), publication.target_offset, publication.count,
        publication.target_stride, publication.element_bytes);
    if (!selected || !target ||
        source.footprint.bytes != target->footprint.bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t bytes = source.footprint.bytes;
    if (bytes == 0u) {
      continue;
    }
    if (source.data == nullptr || target->data == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (target->footprint.dense()) {
      std::memcpy(target->data, source.data, bytes);
      continue;
    }

    const std::byte *from = source.data;
    std::byte *to = target->data;
    for (std::size_t remaining = publication.count; remaining > 1u;
         --remaining) {
      std::memcpy(to, from, source.footprint.width);
      from += source.footprint.stride;
      to += target->footprint.stride;
    }
    std::memcpy(to, from, source.footprint.width);
  }
  return Status::success();
}

} // namespace rund::compute::detail
