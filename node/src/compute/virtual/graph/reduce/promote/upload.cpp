#include "internal.hpp"

#include "../../../../backend.hpp"

#include "../../../../device/residency/pool.hpp"
#include "../../../../pipeline/local.hpp"

#include <kernel/core/checked.hpp>

#include <limits>

namespace rund::compute::detail::graph_reduce::promote_detail {

bool project_upload(PipelineState &pipeline, const VirtualRunProjection &run,
                    const Projection &projection,
                    UploadProjection &result) noexcept {
  result = {};
  if (pipeline.device == nullptr || pipeline.device->backend == Backend::Cpu ||
      pipeline.device->ops == nullptr ||
      pipeline.device->ops->upload_batch == nullptr ||
      pipeline.residency == nullptr || !pipeline.residency->graph_tiled() ||
      pipeline.residency_pool == nullptr ||
      pipeline.residency_bank >= residency::Pool::BankCount ||
      projection.page_count == 0u ||
      projection.page_count > projection.pages.size()) {
    return false;
  }

  const residency::TiledGraphPlan &graph = pipeline.residency->tiled_graph();
  const std::uint32_t bank = pipeline.residency_bank;
  for (std::size_t index = 0u; index < projection.page_count; ++index) {
    const residency::execution::GraphPromotePage promoted =
        projection.promoted[index];
    const residency::PrefetchedPage source = projection.pages[index];
    const residency::execution::GraphPromoteCompletion completion =
        projection.completions[index];
    const residency::TiledGraphResource *const declared =
        graph.resource(promoted.use.resource);
    const residency::PoolPhysicalOwner *const owner =
        declared == nullptr
            ? nullptr
            : pipeline.residency_pool->graph_owner(declared->physical_id);
    if (declared == nullptr ||
        declared->kind != residency::GraphResourceKind::ExternalInput ||
        declared->persistence != residency::ResourcePersistence::Backing ||
        declared->role != residency::GraphResourceRole::Input ||
        declared->page_bytes != promoted.bytes || owner == nullptr ||
        owner->physical_class.type != declared->type ||
        owner->physical_class.format != declared->format ||
        owner->physical_class.page_bytes != declared->page_bytes ||
        owner->physical_class.role != residency::FrameRole::Input ||
        owner->buffers[bank] == nullptr ||
        owner->buffers[bank]->device != pipeline.device ||
        promoted.key != source.key || promoted.key != completion.key ||
        promoted.source_frame != source.physical_frame ||
        promoted.source_frame != completion.source_frame ||
        promoted.target_frame != completion.target_frame ||
        promoted.bytes != completion.bytes || source.frame == nullptr ||
        source.bytes == 0u || source.bytes > promoted.bytes ||
        source.target_offset > promoted.bytes - source.bytes ||
        virtual_host_input_frame(run, source.physical_frame) != source.frame ||
        promoted.bytes > std::numeric_limits<std::size_t>::max()) {
      return false;
    }

    const residency::FrameRegion region = owner->cache_regions[bank];
    if (region.tier != residency::FrameTier::Device ||
        region.role != residency::FrameRole::Input || region.count == 0u ||
        promoted.target_frame < region.first ||
        promoted.target_frame - region.first >= region.count) {
      return false;
    }
    std::uint64_t offset = 0u;
    if (!kernel::checked::mul(promoted.target_frame - region.first,
                              promoted.bytes, offset) ||
        offset > std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    BufferState &buffer = *owner->buffers[bank];
    const std::size_t request_bytes = static_cast<std::size_t>(promoted.bytes);
    const std::size_t request_offset = static_cast<std::size_t>(offset);
    if (request_offset > buffer.bytes ||
        request_bytes > buffer.bytes - request_offset ||
        result.bytes >
            std::numeric_limits<std::uint64_t>::max() - promoted.bytes) {
      return false;
    }
    result.requests[index] = UploadRequest{.buffer = &buffer,
                                           .data = source.frame,
                                           .bytes = request_bytes,
                                           .offset = request_offset};
    result.bytes += promoted.bytes;
  }
  result.request_count = projection.page_count;
  return result.bytes != 0u;
}

} // namespace rund::compute::detail::graph_reduce::promote_detail
