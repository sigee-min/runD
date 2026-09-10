#pragma once

#include "model.hpp"

#include "../../../pipeline/transfer.hpp"
#include "../../run/cache.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace rund::compute::detail::graph_reduce {

[[nodiscard]] Status fold_stage(Stats &, const std::shared_ptr<PipelineState> &,
                                std::uint64_t) noexcept;
[[nodiscard]] PipelineFrameDownloadResult
download_frames(PipelineState &,
                std::span<const PipelineFrameDownload>) noexcept;
[[nodiscard]] Status
write_controls(PipelineState &, const VirtualRunProjection &,
               std::span<const residency::CacheBinding>, residency::FrameRegion,
               Stats &, VirtualTransferInterval *interval = nullptr) noexcept;
[[nodiscard]] Status relocate_graph_lease(PipelineState &,
                                          const residency::TiledGraphPlan &,
                                          residency::Pool &,
                                          residency::EpochLease,
                                          Stats &) noexcept;

} // namespace rund::compute::detail::graph_reduce
