#include "internal.hpp"

#include "../../../device/residency/registry/transaction_owner.hpp"
#include "../../backing.hpp"

#include <array>
#include <span>
#include <utility>

namespace rund::compute::detail {

void abort_virtual_run_transaction(
    VirtualPipelineState &state, VirtualRunTransaction &transaction,
    const VirtualRunWriteCertainty certainty,
    VirtualRunWriteCertainty &effective_certainty, Status &status,
    bool &poison_pipeline) noexcept {
  effective_certainty = certainty;
  if (!transaction.started || transaction.output == nullptr ||
      transaction.provider == nullptr) {
    return;
  }
  VirtualBacking &output = *transaction.output;
  const bool lease_owner_valid = transaction_detail::valid_output_lease(
      transaction, transaction.authority);
  if (!lease_owner_valid) {
    effective_certainty = VirtualRunWriteCertainty::UnknownMayWrite;
    status = Status::fail(Reason::DeviceLost);
    poison_pipeline = true;
  }
  if (transaction.scan && !transaction.output_lease && lease_owner_valid) {
    std::array<residency::FrameRegion, 4u> regions{};
    std::size_t region_count = 0u;
    const bool bounded =
        transaction.authority != nullptr &&
        transaction_detail::scan_authority(state) == transaction.authority &&
        transaction_detail::transaction_regions(transaction, regions,
                                                region_count) &&
        region_count == 4u;
    const VirtualBackingTransactionSpec spec = transaction.token.spec();
    const bool cleaned =
        bounded &&
        transaction.authority->virtual_transactions()
            .cleanup_virtual_transaction_rows(
                spec.backing_id, spec.base_version,
                transaction.result_identity_hi, transaction.result_identity_lo,
                transaction.page_count, transaction.boundary_page,
                transaction.cache_extent,
                std::span<const residency::FrameRegion>{regions.data(),
                                                        region_count},
                transaction.token.generation());
    if (!cleaned) {
      effective_certainty = VirtualRunWriteCertainty::UnknownMayWrite;
      status = Status::fail(Reason::DeviceLost);
      poison_pipeline = true;
    }
  }
  if (effective_certainty == VirtualRunWriteCertainty::UnknownMayWrite) {
    if (transaction.token) {
      transaction.provider->quarantine_unknown(std::move(transaction.token));
    }
    poison_pipeline = true;
    status = Status::fail(Reason::DeviceLost);
  } else {
    if (transaction.token) {
      transaction.provider->abort_known(std::move(transaction.token));
    }
  }
  if (transaction.output_lease && lease_owner_valid) {
    transaction.output_lease.authority->virtual_transactions()
        .abort_virtual_transaction_lease(
            std::move(transaction.output_lease),
            effective_certainty == VirtualRunWriteCertainty::UnknownMayWrite);
  }
  if (effective_certainty == VirtualRunWriteCertainty::KnownNoWrite) {
    VirtualBackingAccess::clear_recovery(output);
  }
  transaction_detail::clear_transaction(transaction);
}

} // namespace rund::compute::detail
