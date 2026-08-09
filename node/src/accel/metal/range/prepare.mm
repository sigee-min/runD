#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../kernel/backend/run.hpp"
#include "../../kernel/preparation.hpp"
#include "../../kernel/scratch.hpp"
#include "../../range_aggregate/plan.hpp"
#include "../pipeline/template.hpp"
#include "../scratch.hpp"
#include "encode/dispatch.hpp"
#include "local.hpp"
#include "pipeline/store.hpp"
#include "resources/prepare.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace rund::node::accel::detail {

rund::AccelCheck
PrepareMetalRange(const rund::AccelDevice &pick, const RangePlan &range,
                  const MetalRangeBinds &bindings,
                  const BoundControl *const control,
                  std::shared_ptr<void> &resources,
                  const MetalKernelImmutablePipelines *const pipelines) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  resources.reset();
  if (!MetalPickOwnsAdapter(pick)) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (adapter == nullptr || adapter->device == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  SetMetalLastError(*adapter, "ok");
  const std::optional<RangeExec> execution = RangeExec::from(range);
  if (!execution.has_value()) {
    SetMetalLastError(*adapter, "compute_range_aggregate_invalid");
    return rund::AccelCheck{false, "compute_range_aggregate_invalid"};
  }
  for (std::size_t index = 0u; index < range.stage_count(); ++index) {
    if (!execution->stage_dispatch_fits(
            index, std::numeric_limits<std::uint32_t>::max())) {
      SetMetalLastError(*adapter, "compute_dispatch_overflow");
      return rund::AccelCheck{false, "compute_dispatch_overflow"};
    }
  }

  auto *const raw = new MetalRangeResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalRangeResources};
  raw->adapter = adapter;
  raw->range = range;
  raw->stage_count = static_cast<std::uint32_t>(range.stage_count());
  raw->input = bindings.input();
  raw->output = bindings.output();
  rund::AccelCheck check{true, "ok"};
  if (!PrepareMetalRangeTemps(*adapter, *raw)) {
    return rund::AccelCheck{false, MetalLastError(adapter)};
  }
  if (!PrepareMetalRangeControl(pick, control, *raw, pipelines)) {
    check = {false, "compute_range_aggregate_invalid"};
  } else if (pipelines != nullptr &&
             !pipelines->ready(raw->stage_count, raw->controlled)) {
    check = {false, "accel_metal_pipeline_unavailable"};
  } else {
    for (std::size_t index = 0u; check.ok && index < range.stage_count();
         ++index) {
      std::shared_ptr<void> pipeline;
      if (pipelines != nullptr) {
        pipeline = pipelines->stages[index];
        check =
            AssessMetalRange(*adapter, *execution, pipeline) ==
                    MetalRangeAssessment::Ready
                ? rund::AccelCheck{true, "ok"}
                : rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
      } else {
        const MetalRangeAttempt attempt =
            CompileMetalRange(*adapter, *execution, pipeline);
        check = attempt.status == MetalRangeAttemptStatus::Ready
                    ? rund::AccelCheck{true, "ok"}
                    : rund::AccelCheck{false, attempt.reason};
      }
      if (check.ok) {
        raw->pipelines[index] = std::move(pipeline);
      }
    }
  }
  if (!check.ok) {
    SetMetalLastError(*adapter, check.reason);
    return check;
  }
  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  (void)bindings;
  (void)range;
  (void)control;
  (void)resources;
  (void)pipelines;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
