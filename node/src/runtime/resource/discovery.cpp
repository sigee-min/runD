#include "discovery.hpp"

#include <kernel/dispatch/kernel.hpp>

#include <utility>

namespace rund::node::runtime_detail::resource {
namespace {

[[nodiscard]] ::rund::EvidenceTruth
Evidence(const kernel::WorkerTruthLevel truth) noexcept {
  switch (truth) {
  case kernel::WorkerTruthLevel::Verified:
    return ::rund::EvidenceTruth::Verified;
  case kernel::WorkerTruthLevel::HintOnly:
    return ::rund::EvidenceTruth::Hint;
  case kernel::WorkerTruthLevel::Unknown:
    return ::rund::EvidenceTruth::Unknown;
  }
  return ::rund::EvidenceTruth::Unknown;
}

[[nodiscard]] Result Describe(BackendSelection selection) {
  if (!selection) {
    return Result{.code = selection.code};
  }

  const std::uint32_t width = selection.requested_worker_width;
  const kernel::WorkerBackendCapabilities capabilities =
      kernel::InspectWorkerBackend(selection.backend, width);

  Result out{};
  out.code = ReasonCode::Ok;
  out.resources.observed.code = ReasonCode::Ok;
  out.resources.observed.workers = width;
  out.resources.worker_backend = selection.backend;
  if (capabilities.worker_capacity_milli != nullptr &&
      capabilities.worker_capacity_count != 0u) {
    out.resources.observed.worker_capacity_milli.assign(
        capabilities.worker_capacity_milli,
        capabilities.worker_capacity_milli +
            capabilities.worker_capacity_count);
  }

  out.resources.observed.topology = ::rund::Topology{
      .numa = selection.verified_numa ? ::rund::EvidenceTruth::Verified
                                      : ::rund::EvidenceTruth::Unknown,
      .affinity = Evidence(capabilities.affinity_truth_level),
      .worker_capacity = Evidence(capabilities.worker_capacity_truth_level),
      .numa_domains = selection.verified_numa ? 1u : 0u,
      .spans_numa_domains = false,
  };
  return out;
}

[[nodiscard]] ReasonCode Admit(const ResourceEnvelope &resources,
                               const Request &request) noexcept {
  const ::rund::Resources &observed = resources.observed;
  if (observed.workers == 0u) {
    return ReasonCode::BackendWidthRequired;
  }
  if (!resources.worker_backend) {
    return ReasonCode::RuntimeResourcesInvalid;
  }
  if (request.require_verified_numa &&
      observed.topology.numa != ::rund::EvidenceTruth::Verified) {
    return ReasonCode::VerifiedTopologyRequired;
  }
  if (request.require_verified_affinity &&
      observed.topology.affinity != ::rund::EvidenceTruth::Verified) {
    return ReasonCode::AffinityTruthUnavailable;
  }
  if (observed.topology.worker_capacity == ::rund::EvidenceTruth::Verified) {
    if (observed.worker_capacity_milli.size() != observed.workers) {
      return ReasonCode::WorkerCapacityTruthUnavailable;
    }
    for (const std::uint32_t capacity : observed.worker_capacity_milli) {
      if (capacity == 0u) {
        return ReasonCode::WorkerCapacityTruthUnavailable;
      }
    }
  } else if (request.require_verified_capacity) {
    return ReasonCode::WorkerCapacityTruthUnavailable;
  }
  return ReasonCode::Ok;
}

} // namespace

Result Resolve(BackendSelection selection, const Request &request) {
  Result result = Describe(std::move(selection));
  if (result) {
    result.code = Admit(result.resources, request);
  }
  return result;
}

} // namespace rund::node::runtime_detail::resource
