#include "../internal.hpp"

#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"
#include "src/compute/virtual/state.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace rund::measure::compute::route_matrix {
namespace {

using Status = ::rund::compute::Status;
using Reason = ::rund::compute::Reason;
using Access = ::rund::compute::detail::VirtualBackingAccess;

[[nodiscard]] std::string error_text(const Reason reason) {
  const auto text = Status::fail(reason).error();
  return std::string{text};
}

[[nodiscard]] std::uint64_t owner_id(const Job &job) noexcept {
  if (job.pipeline == nullptr) {
    return 0u;
  }
  const auto state =
      ::rund::compute::detail::VirtualPipelineAccess::state(*job.pipeline);
  return state == nullptr || state->device_vsm_product_cache == nullptr
             ? 0u
             : static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(
                   state->device_vsm_product_cache.get()));
}

} // namespace

PairEvidence run_pair(Job &cpu, Job &selected, const CaseSpec &spec,
                      const std::uint64_t expected_hash,
                      std::vector<std::int32_t> &cpu_output,
                      std::vector<std::int32_t> &selected_output,
                      const bool cold) noexcept {
  PairEvidence result{};
  const auto begin = Clock::now();
  result.cpu = run_one(cpu, spec, &cpu_output, true, cold);
  result.selected = run_one(selected, spec, &selected_output, true, cold);
  const auto end = Clock::now();
  if (cold) {
    result.total_us = micros(end - begin);
  }
  result.cpu_output_ok =
      result.cpu.ok && oracle::output_ok(cpu_output, spec, expected_hash);
  result.selected_output_ok =
      result.selected.ok &&
      oracle::output_ok(selected_output, spec, expected_hash);
  result.cpu_output_hash = oracle::hash_values(cpu_output);
  result.selected_output_hash = oracle::hash_values(selected_output);
  return result;
}

void hide_row_timing(Row &row, const char *const authority) noexcept {
  row.cold = {};
  row.cpu_warm = {};
  row.backend_warm = {};
  row.gpu_kernel_ns = 0u;
  row.submit_span_ns = 0u;
  row.timing_valid = false;
  row.timing_authority = authority;
}

void capture_route(Row &row, RouteObserver *const observer) noexcept {
  if (observer == nullptr) {
    return;
  }
  observer->finish();
  row.route_evidence = observer->evidence();
  row.proof_hi = row.route_evidence.proof_hi;
  row.proof_lo = row.route_evidence.proof_lo;
}

void fail_row(Row &row, RouteObserver *const observer,
              const std::string_view status, const Reason reason,
              const char *const authority, const char *const detail) {
  capture_route(row, observer);
  row.status = status;
  row.reason_code = static_cast<std::uint64_t>(reason);
  row.reason = detail == nullptr ? error_text(reason) : std::string{detail};
  hide_row_timing(row, authority);
}

bool publish_evidence(Row &row, Job &cpu, Job &selected,
                      RouteObserver *const observer,
                      const std::uint64_t expected_count,
                      const std::uint64_t version_before,
                      std::vector<std::int32_t> &cpu_output,
                      std::vector<std::int32_t> &selected_output,
                      WarmCohort &cohort,
                      const ::rund::compute::Stats &cpu_stats,
                      const ::rund::compute::Stats &selected_stats) {
  const bool cpu_read = read_output(cpu, expected_count, cpu_output);
  const bool selected_read =
      read_output(selected, expected_count, selected_output);
  row.cpu_ok =
      cpu_read && oracle::output_ok(cpu_output, row.spec, row.expected_hash);
  row.output_ok = selected_read && oracle::output_ok(selected_output, row.spec,
                                                     row.expected_hash);
  row.cpu_output_hash = oracle::hash_values(cpu_output);
  row.output_hash = oracle::hash_values(selected_output);
  row.version_before = version_before;
  row.version_after = Access::version(*selected.output);
  row.cpu_command_submits = cpu_stats.command_submits;
  row.cpu_dispatches = cpu_stats.dispatches;
  row.cpu_warm = summarize(cohort.cpu_values, expected_count, true);
  row.backend_warm = summarize(cohort.selected_values, expected_count, true);
  row.stats = selected_stats;
  row.residency = selected_stats.pipeline.residency;
  capture_route(row, observer);
  row.owner_id = owner_id(selected);
  row.route =
      oracle::route_name(selected_stats, row.route_evidence, row.owner_id);
  row.owner_stable = row.route_evidence.owner_stable;
  row.backing_match = oracle::backing_ok(row);
  row.one_final = oracle::cohort_ok(row);
  row.allocation_free_60 = row.residency.samples_allocation_free(WarmSamples);
  row.timing_valid = row.cpu_warm.complete && row.backend_warm.complete;
  row.gpu_kernel_ns = 0u;
  row.submit_span_ns = 0u;
  row.status = "ok";
  row.timing_authority = "publishable";
  row.reason_code = static_cast<std::uint64_t>(Reason::Ok);
  row.reason = "";
  const char *const mismatch = oracle::mismatch_reason(row);
  row.comparable = mismatch == nullptr;
  if (!row.comparable) {
    row.status = "not_comparable";
    row.reason = mismatch == nullptr ? "route_or_counter_mismatch" : mismatch;
    row.label = "indeterminate";
    hide_row_timing(row, "unavailable_not_comparable");
  } else {
    row.label = oracle::label(row.cpu_warm, row.backend_warm);
  }
  return row.status == "ok" || row.status == "not_comparable";
}

} // namespace rund::measure::compute::route_matrix
