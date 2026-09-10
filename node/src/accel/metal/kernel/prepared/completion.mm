#include "local.hpp"

#include "../../pipeline/named.hpp"
#include "../trace/encoder.hpp"
#include "../trace/replay.hpp"

#include "../../../kernel/reset/stats.hpp"

#include <rund/counter.hpp>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

void CompleteMetalPreparedRun(void *const raw, KernelResult submitted,
                              const bool trace) noexcept {
  auto *const state =
      static_cast<submission::State<MetalKernelResources> *>(raw);
  if (state == nullptr) {
    return;
  }
  const submission::Claim<MetalKernelResources> claim =
      submission::Take(*state);
  if (!claim) {
    return;
  }
  MetalKernelResources &resources = *claim.owner;
  if (trace) {
    if (submitted.check.ok) {
      submitted.check = FoldMetalDispatchTrace(
          resources.trace,
          (__bridge id<MTLDevice>)resources.adapter->device.get(),
          submitted.stats);
      if (submitted.check.ok && resources.adapter != nullptr) {
        RecordMetalDispatchTrace(
            *resources.adapter, submitted.stats.run.time.accel_kernel_ns,
            submitted.stats.run.time.accel_timestamp_count);
      }
    }
  }
  if (submitted.check.ok && resources.adapter != nullptr) {
    submitted.check =
        FinishMetalSteps(*resources.adapter, resources, &submitted.stats);
  }
  submitted.stats.run.work.dispatch_count =
      submitted.check.ok ? resources.dispatch_count : 0u;
  SetResetStats(submitted.stats, submitted.check.ok, resources.reset_count,
                resources.reset_bytes);
  claim.completion(claim.user, submitted);
}

} // namespace

void CompleteMetalPrepared(void *const raw, KernelResult submitted) noexcept {
  CompleteMetalPreparedRun(raw, submitted, false);
}

void CompleteMetalPreparedTrace(void *const raw,
                                KernelResult submitted) noexcept {
  CompleteMetalPreparedRun(raw, submitted, true);
}

#endif

} // namespace rund::node::accel::detail
