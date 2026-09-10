#include "internal.hpp"

#include "../../../device/residency/registry/transaction_owner.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <span>

namespace rund::compute::detail {

Status
prepare_virtual_run_transaction(VirtualPipelineState &state,
                                VirtualRunTransaction &transaction) noexcept {
  if (!transaction.started) {
    return Status::success();
  }
  const Status base =
      transaction_detail::validate_transaction_base(state, transaction);
  if (!base) {
    return base;
  }
  return transaction.provider->prepare_commit(transaction.token);
}

Status prepare_virtual_run_transaction_rows(VirtualPipelineState &state,
                                            VirtualRunTransaction &transaction,
                                            const bool require_rows) noexcept {
  if (!transaction.started) {
    return require_rows ? Status::fail(Reason::PipelineInvalid)
                        : Status::success();
  }
  if (transaction.scan && transaction.journal == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (!transaction.scan || !transaction.rows_active) {
    return require_rows ? Status::fail(Reason::PipelineInvalid)
                        : Status::success();
  }
  const Status base =
      transaction_detail::validate_transaction_base(state, transaction);
  if (!base) {
    return base;
  }
  if (transaction.output_lease) {
    return Status::success();
  }
  if (!transaction.cursor.active || transaction.authority == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::array<residency::FrameRegion, 4u> regions{};
  std::size_t count = 0u;
  if (!transaction_detail::transaction_regions(transaction, regions, count) ||
      count != 4u) {
    return Status::fail(Reason::CompletionInvalid);
  }
  std::size_t expected_rows = 0u;
  if (require_rows) {
    const std::uint64_t capacity = transaction.host_output_frame_capacity;
    if (capacity > std::numeric_limits<std::uint64_t>::max() / 2u) {
      return Status::fail(Reason::PipelineCapacity);
    }
    const std::uint64_t expected =
        std::min(transaction.page_count, capacity * 2u);
    if (expected == 0u ||
        expected > static_cast<std::uint64_t>(
                       std::numeric_limits<std::size_t>::max())) {
      return Status::fail(Reason::PipelineCapacity);
    }
    expected_rows = static_cast<std::size_t>(expected);
  }
  const VirtualBackingTransactionSpec spec = transaction.token.spec();
  if (!transaction.authority->virtual_transactions()
           .prepare_virtual_transaction_lease(
               spec.backing_id, spec.base_version,
               transaction.result_identity_hi, transaction.result_identity_lo,
               transaction.page_count, transaction.boundary_page,
               transaction.cache_extent,
               std::span<const residency::FrameRegion>{regions.data(), count},
               std::span<const residency::VirtualTransactionLease::Row>{
                   transaction.journal->data(), transaction.journal_count},
               transaction.token.generation(), expected_rows, require_rows,
               transaction.output_lease)) {
    return Status::fail(Reason::CompletionInvalid);
  }
  return Status::success();
}

} // namespace rund::compute::detail
