#include "internal.hpp"

#include "../../backing.hpp"
#include "../../state.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <utility>

namespace rund::compute::detail {

Status begin_virtual_run_transaction(
    VirtualPipelineState &state, const VirtualRunAdmission &admission,
    const VirtualRunProjection &run, VirtualBacking &output,
    const std::uint64_t page_count,
    VirtualRunTransaction &transaction) noexcept {
  transaction = VirtualRunTransaction{};
  if (!admission.transaction_window && !admission.transaction_scan) {
    return Status::success();
  }

  VirtualBackingTransaction *const provider =
      dynamic_cast<VirtualBackingTransaction *>(&output);
  if (provider == nullptr) {
    return Status::success();
  }
  residency::Authority *bound_authority = nullptr;
  if (admission.transaction_scan) {
    bound_authority = transaction_detail::scan_authority(state);
    if (bound_authority == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  auto journal = std::unique_ptr<VirtualRunTransaction::Journal>{
      new (std::nothrow) VirtualRunTransaction::Journal{}};
  if (journal == nullptr) {
    return Status::fail(Reason::PipelineCapacity);
  }
  transaction.journal = std::move(journal);
  transaction.output = &output;
  transaction.provider = provider;
  transaction.authority = bound_authority;
  transaction.scan = admission.transaction_scan;
  transaction.result_identity_hi = run.result_identity_hi;
  transaction.result_identity_lo = run.result_identity_lo;
  transaction.output_regions = run.output_regions;
  transaction.cache_extent = run.cache_extent;
  transaction.page_count = page_count;
  transaction.host_output_frame_capacity = run.host_output_frame_capacity;

  const std::uint64_t backing_page_bytes = run.output_page_bytes;
  const std::uint64_t host_output_capacity = run.host_output_frame_capacity;
  const std::uint64_t host_output_first = run.first_host_output_frame;
  const bool host_output_span_valid =
      host_output_capacity != 0u &&
      host_output_capacity <= std::numeric_limits<std::uint64_t>::max() / 2u &&
      host_output_first <= std::numeric_limits<std::uint64_t>::max() -
                               host_output_capacity * 2u &&
      host_output_first + host_output_capacity * 2u <=
          std::numeric_limits<std::uint32_t>::max();
  const bool output_regions_valid =
      transaction_detail::valid_captured_region(transaction.output_regions[0u],
                                                residency::FrameTier::Device) &&
      transaction_detail::valid_captured_region(transaction.output_regions[1u],
                                                residency::FrameTier::Device) &&
      transaction_detail::disjoint_captured_regions(transaction.output_regions);
  if (state.pipeline == nullptr ||
      (transaction.scan && state.pipeline->residency_pool == nullptr) ||
      backing_page_bytes == 0u || run.output_payload_bytes == 0u ||
      run.active.output_bytes == 0u ||
      run.active.output_bytes != output.size_bytes() || page_count == 0u ||
      (transaction.scan &&
       (!output_regions_valid || !host_output_span_valid))) {
    transaction = VirtualRunTransaction{};
    return Status::fail(Reason::PipelineInvalid);
  }
  if (transaction.scan) {
    const std::uint32_t host_first = run.first_host_output_frame;
    const std::uint32_t host_capacity = run.host_output_frame_capacity;
    transaction.host_output_regions = {
        residency::FrameRegion{.tier = residency::FrameTier::Host,
                               .role = residency::FrameRole::Output,
                               .first = host_first,
                               .count = host_capacity},
        residency::FrameRegion{
            .tier = residency::FrameTier::Host,
            .role = residency::FrameRole::Output,
            .first = static_cast<std::uint32_t>(host_first + host_capacity),
            .count = host_capacity}};
    if (!transaction_detail::disjoint_captured_regions(
            transaction.host_output_regions)) {
      transaction = VirtualRunTransaction{};
      return Status::fail(Reason::PipelineInvalid);
    }
    transaction.boundary_page =
        run.active.output_bytes % run.output_payload_bytes == 0u
            ? std::numeric_limits<std::uint64_t>::max()
            : page_count - 1u;
  }
  const std::uint64_t backing_page_count =
      run.active.output_bytes / backing_page_bytes +
      (run.active.output_bytes % backing_page_bytes != 0u);
  const VirtualBackingTransactionSpec spec{
      .backing_id = VirtualBackingAccess::id(output),
      .base_version = VirtualBackingAccess::version(output),
      .logical_bytes = run.active.output_bytes,
      .backing_page_bytes = backing_page_bytes,
      .backing_page_count = backing_page_count,
      .write_page_bytes = run.output_payload_bytes,
      .write_page_count = page_count,
  };
  VirtualBackingAccess::require_recovery(output, run.active.output_bytes);
  const Status begun = provider->begin(spec, transaction.token);
  if (!begun) {
    VirtualBackingAccess::clear_recovery(output);
    transaction = VirtualRunTransaction{};
    if (begun.reason() == Reason::BackendUnsupported) {
      return Status::success();
    }
    return begun;
  }
  if (!transaction.scan) {
    VirtualBackingAccess::bind_transaction(output, provider,
                                           &transaction.token);
  }
  transaction.started = true;
  return Status::success();
}

} // namespace rund::compute::detail
