#pragma once

#include "common.hpp"

#include "../../../compact/local.hpp"
#include "../../../gather/local.hpp"
#include "../../../histogram/local.hpp"
#include "../../../scatter/local.hpp"

#include <limits>

namespace rund::node::accel::detail {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] inline bool
DescribeMetalCompactPipelineStatus(const std::shared_ptr<void> &resources,
                                   MetalPipelineStatusBindings &out) noexcept {
  using namespace metal_pipeline_status;
  out = {};
  const auto *const compact =
      static_cast<const MetalCompactEncodeResources *>(resources.get());
  if (compact == nullptr ||
      !append(
          out,
          binding(compact->scan_status, MetalPipelineStatusEncoding::BitFlags,
                  {reason(rund::compute::Reason::ScanSumOverflow),
                   reason(rund::compute::Reason::BoundedCountInvalid), 0u}))) {
    return false;
  }
  return compact->plan.status_bytes == 0u ||
         append(out,
                binding(
                    compact->status, MetalPipelineStatusEncoding::Limit,
                    {reason(rund::compute::Reason::CompactCapacityInsufficient),
                     0u, 0u},
                    0u, compact->plan.output_capacity));
}

[[nodiscard]] inline bool
DescribeMetalGatherPipelineStatus(const std::shared_ptr<void> &resources,
                                  MetalPipelineStatusBindings &out) noexcept {
  using namespace metal_pipeline_status;
  out = {};
  const auto *const gather =
      static_cast<const MetalGatherEncodeResources *>(resources.get());
  if (gather == nullptr) {
    return false;
  }
  MetalPipelineStatusBinding described =
      binding(gather->status, MetalPipelineStatusEncoding::Mapping,
              {reason(rund::compute::Reason::BoundedCountInvalid),
               reason(rund::compute::Reason::GatherIndexOutOfRange), 0u, 0u},
              0u, gather->plan.element_count);
  described.indirect_dispatch_count = 1u;
  described.telemetry = MetalPipelineStatusTelemetry::Gather;
  return append(out, described);
}

[[nodiscard]] inline bool DescribeMetalHistogramPipelineStatus(
    const std::shared_ptr<void> &resources,
    MetalPipelineStatusBindings &out) noexcept {
  using namespace metal_pipeline_status;
  out = {};
  const auto *const histogram =
      static_cast<const MetalHistogramEncodeResources *>(resources.get());
  return histogram != nullptr &&
         append(out,
                binding(histogram->status,
                        MetalPipelineStatusEncoding::Sentinel,
                        {reason(rund::compute::Reason::HistogramBinInvalid), 0u,
                         0u},
                        std::numeric_limits<std::uint32_t>::max()));
}

[[nodiscard]] inline bool
DescribeMetalScatterPipelineStatus(const std::shared_ptr<void> &resources,
                                   MetalPipelineStatusBindings &out) noexcept {
  using namespace metal_pipeline_status;
  out = {};
  const auto *const scatter =
      static_cast<const MetalScatterEncodeResources *>(resources.get());
  return scatter != nullptr &&
         append(out,
                binding(scatter->status, MetalPipelineStatusEncoding::Scatter,
                        {reason(rund::compute::Reason::ScatterIndexOutOfRange),
                         reason(rund::compute::Reason::ScatterDuplicateIndex),
                         reason(rund::compute::Reason::ScatterInvalid)},
                        std::numeric_limits<std::uint32_t>::max()));
}

#endif
} // namespace rund::node::accel::detail
