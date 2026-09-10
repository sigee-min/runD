#pragma once

#include "admission.hpp"
#include "backing_set.hpp"
#include "device_vsm/route.hpp"
#include "execution.hpp"
#include "reduce.hpp"
#include "scan.hpp"
#include "transaction.hpp"

#include <rund/compute/stats.hpp>

#include <span>

namespace rund::compute::detail {

struct VirtualRunDispatchResult final {
  Status status{Status::fail(Reason::BackendUnsupported)};
  std::uint64_t failed_page{ResidencyStats::no_failed_page};
  std::uint64_t output_hash{};
  bool poison_pipeline{};
  bool selected{};
  bool direct_terminal{};
  bool empty{};
  bool graph_execution{};
  bool reduction_pending{};
  VirtualRunWriteCertainty certainty{VirtualRunWriteCertainty::KnownNoWrite};
};

// Work is prepared by the dispatch owner and remains alive through the shared
// terminal. The prepared bit makes reduction/scan setup a single invocation
// transition even when VSM declines and ordinary execution follows.
struct VirtualRunWork final {
  VirtualReduction reduction{};
  VirtualScan scan{};
  Status prepared_status{Status::success()};
  bool prepared{};
};

[[nodiscard]] VirtualRunDispatchResult
dispatch_result(const VirtualExecutionResult &) noexcept;

[[nodiscard]] VirtualRunDispatchResult
dispose_device_vsm_route(VirtualPipelineState &, VirtualRunResources &,
                         VirtualDeviceVsmPostStage &, VirtualDeviceVsmScope,
                         Status, bool &) noexcept;

[[nodiscard]] VirtualRunDispatchResult
execute_device_vsm_stage(VirtualPipelineState &,
                         std::span<VirtualBacking *const>, VirtualBacking &,
                         const VirtualRunProjection &,
                         VirtualDeviceVsmPostStage &, VirtualDeviceVsmScope,
                         Stats &, VirtualRunResources &, bool &) noexcept;

[[nodiscard]] Status prepare_work(const VirtualRunProjection &,
                                  VirtualRunWork &) noexcept;

[[nodiscard]] VirtualRunDispatchResult dispatch_virtual_route(
    VirtualPipelineState &, std::span<VirtualBacking *const>, VirtualBacking &,
    VirtualBacking &, VirtualRunProjection &,
    const VirtualDeviceVsmCandidate &, std::uint64_t, Stats &,
    VirtualRunWork &, VirtualRunTransaction &, VirtualRunResources &,
    bool &) noexcept;

[[nodiscard]] VirtualRunDispatchResult
dispatch_run(VirtualPipelineState &, std::span<VirtualBacking *const>,
             VirtualBacking &, VirtualBacking &, const VirtualRunProjection &,
             const VirtualRunAdmission &, Stats &, VirtualRunWork &,
             VirtualRunTransaction *transaction = nullptr) noexcept;

} // namespace rund::compute::detail
