#include "internal.hpp"

#include "src/compute/virtual/backing.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <string_view>
#include <vector>

namespace rund::measure::compute::route_matrix {
namespace {

using Reason = ::rund::compute::Reason;
using Access = ::rund::compute::detail::VirtualBackingAccess;

[[nodiscard]] bool run_case(const ComputeBackend backend,
                            const ProfileMode profile, const CaseSpec &spec,
                            Row &row) noexcept {
  row.spec = spec;
  row.profile = profile;
  row.backend = backend;
  row.expected_route =
      spec.family == "pointwise_callback" || spec.family == "pointwise_resident"
          ? "device_vsm"
      : spec.family == "spatial_window" ? "device_vsm"
                                        : "natural";
  row.elements = elements(spec);
  row.page_count = pages(row.elements);
  row.expected_hash = 0u;

  if (!spec.supported) {
    fail_row(row, nullptr, "unsupported", Reason::BackendUnsupported,
             "unavailable_unsupported", "no_natural_staged_distinction");
    return true;
  }
  try {
    auto device = ::rund::compute::open(TargetFor(backend));
    if (!device) {
      fail_row(row, nullptr, "unavailable", device.reason(),
               "unavailable_device");
      return false;
    }
    // Device owns the DeviceState observed by RouteObserver. Declaration
    // order makes the observer restore the table before Device is destroyed.
    std::unique_ptr<RouteObserver> observer;
    auto info = device->info();
    if (!info || info->backend != backend) {
      const Reason reason = info ? Reason::BackendMismatch : info.reason();
      fail_row(row, observer.get(), "unavailable", reason,
               "unavailable_backend");
      return false;
    }
    row.api = backend == ComputeBackend::Vulkan
                  ? "vulkan/MoltenVK_portability_only"
                  : "metal/native";
    row.driver = info->driver;
    row.driver_details = info->driver_details;
    auto cpu_device = ::rund::compute::open(::rund::compute::Target::cpu());
    if (!cpu_device) {
      fail_row(row, observer.get(), "unavailable", cpu_device.reason(),
               "unavailable_cpu");
      return false;
    }
    if (backend != ComputeBackend::Cpu) {
      observer = std::make_unique<RouteObserver>(*device);
    }
    Job cpu = prepare_job(*cpu_device, spec, false, {}, false);
    Job selected =
        prepare_job(*device, spec, spec.resident, residency_config(spec), true);
    if (!cpu.prepared || !selected.prepared) {
      const bool cpu_failed = !cpu.prepared;
      const bool selected_failed = !selected.prepared;
      const Reason reason = selected_failed ? selected.reason : cpu.reason;
      const bool both_failed = cpu_failed && selected_failed;
      fail_row(row, observer.get(),
               both_failed || selected_failed ? "prepare_failed"
                                              : "unavailable",
               reason,
               both_failed       ? "unavailable_cpu_and_selected_prepare"
               : selected_failed ? "unavailable_selected_prepare"
                                 : "unavailable_cpu_prepare");
      return false;
    }
    row.plan_hi = selected.plan.residency.identity_hi;
    row.plan_lo = selected.plan.residency.identity_lo;
    row.semantic_hi = cpu.plan.residency.identity_hi;
    row.semantic_lo = cpu.plan.residency.identity_lo;
    const std::uint64_t expected_count = elements(spec);
    std::vector<std::int32_t> expected(
        static_cast<std::size_t>(expected_count));
    oracle::fill_expected(expected, spec);
    row.expected_hash = oracle::hash_values(expected);
    std::vector<std::int32_t> cpu_output(
        static_cast<std::size_t>(expected_count));
    std::vector<std::int32_t> selected_output(
        static_cast<std::size_t>(expected_count));
    const std::uint64_t version_before = Access::version(*selected.output);
    const PairEvidence cold = run_pair(cpu, selected, spec, row.expected_hash,
                                       cpu_output, selected_output, true);
    row.cold = selected.cold;
    row.cold.run_us = cold.selected.wall_us;
    row.cold.read_us = 0.0;
    row.cold.total_us = cold.total_us;
    row.cpu_ok = cold.cpu_output_ok;
    row.output_ok = cold.selected_output_ok;
    row.cpu_output_hash = cold.cpu_output_hash;
    row.output_hash = cold.selected_output_hash;
    if (!cold.cpu.ok || !cold.selected.ok || !row.cpu_ok || !row.output_ok) {
      const Reason reason = !cold.selected.ok ? cold.selected.reason
                            : !cold.cpu.ok    ? cold.cpu.reason
                                              : Reason::BackendFailed;
      fail_row(row, observer.get(), "failed", reason, "unavailable_cold");
      return false;
    }
    const PairEvidence condition =
        run_pair(cpu, selected, spec, row.expected_hash, cpu_output,
                 selected_output, false);
    if (!condition.cpu_output_ok || !condition.selected_output_ok) {
      const Reason reason = !condition.selected.ok ? condition.selected.reason
                            : !condition.cpu.ok    ? condition.cpu.reason
                                                   : Reason::BackendFailed;
      fail_row(row, observer.get(), "failed", reason,
               "unavailable_conditioning");
      return false;
    }
    WarmCohort cohort =
        run_warm(cpu, selected, spec, expected_count, row.expected_hash,
                 cpu_output, selected_output);
    const auto cpu_stats = cpu.pipeline->stats();
    const auto selected_stats = selected.pipeline->stats();
    if (!cohort.ok) {
      fail_row(row, observer.get(), "failed", cohort.reason,
               "unavailable_samples");
      return false;
    }
    return publish_evidence(row, cpu, selected, observer.get(), expected_count,
                            version_before, cpu_output, selected_output, cohort,
                            cpu_stats, selected_stats);
  } catch (const std::bad_alloc &) {
    fail_row(row, nullptr, "failed", Reason::BufferCapacity,
             "unavailable_allocation");
    return false;
  }
}

[[nodiscard]] std::vector<CaseSpec> cases(const ProfileMode profile) {
  std::vector<CaseSpec> result;
  const auto add_pointwise = [&](const std::uint64_t q) {
    result.push_back({"pointwise_callback", "memory", q, 0u, false, true});
    result.push_back({"pointwise_staged", "memory", q, 0u, false, false});
    result.push_back({"pointwise_resident", "resident", q, 0u, true, true});
  };
  const auto add_window = [&](const std::uint64_t q,
                              const std::uint64_t width) {
    result.push_back({"spatial_window", "memory", q, width, false, true});
  };
  if (profile == ProfileMode::Core) {
    add_pointwise(9u);
    add_window(9u, 2u);
    return result;
  }
  for (const std::uint64_t q : {3u, 9u, 257u, 4096u}) {
    add_pointwise(q);
    add_window(q, 2u);
  }
  add_window(9u, 4u);
  add_window(3u, 4u);
  return result;
}

} // namespace

bool parse_backend(const std::string_view value,
                   ComputeBackend &backend) noexcept {
  if (value == "metal") {
    backend = ComputeBackend::Metal;
    return true;
  }
  if (value == "vulkan") {
    backend = ComputeBackend::Vulkan;
    return true;
  }
  return false;
}

bool parse_profile(const std::string_view value,
                   ProfileMode &profile) noexcept {
  if (value == "core") {
    profile = ProfileMode::Core;
    return true;
  }
  if (value == "full") {
    profile = ProfileMode::Full;
    return true;
  }
  return false;
}

const char *profile_name(const ProfileMode profile) noexcept {
  return profile == ProfileMode::Core ? "core" : "full";
}

bool run_matrix(const ComputeBackend backend, const ProfileMode profile) {
  for (const CaseSpec &spec : cases(profile)) {
    Row row{};
    (void)run_case(backend, profile, spec, row);
    print_row(row);
  }
  // A row can be unavailable, unsupported, or not comparable while the
  // matrix enumeration itself has completed successfully.
  return true;
}

} // namespace rund::measure::compute::route_matrix
