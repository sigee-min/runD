#include "internal.hpp"

#include "../../../device/residency/registry/transaction_owner.hpp"
#include "../../backing.hpp"

#include <utility>

namespace rund::compute::detail {

Status commit_virtual_run_transaction(
    VirtualPipelineState &state, VirtualRunTransaction &transaction,
    bool &poison_pipeline,
    VirtualRunWriteCertainty &effective_certainty) noexcept {
  if (!transaction.started || transaction.output == nullptr ||
      transaction.provider == nullptr) {
    return Status::success();
  }
  if (transaction.scan) {
    const Status base =
        transaction_detail::validate_transaction_base(state, transaction);
    if (!base) {
      Status failure = base;
      abort_virtual_run_transaction(
          state, transaction, VirtualRunWriteCertainty::UnknownMayWrite,
          effective_certainty, failure, poison_pipeline);
      return failure;
    }
  }
  VirtualBacking &output = *transaction.output;
  VirtualBackingTransaction &provider = *transaction.provider;
  const VirtualBackingTransactionSpec spec = transaction.token.spec();
  const VirtualBackingTransactionResult result =
      provider.commit(transaction.token);
  const auto abort_lease = [&](const bool unknown) noexcept {
    if (transaction.output_lease) {
      transaction.output_lease.authority->virtual_transactions()
          .abort_virtual_transaction_lease(std::move(transaction.output_lease),
                                           unknown);
    }
  };
  switch (result) {
  case VirtualBackingTransactionResult::Success: {
    const bool version_valid =
        VirtualBackingAccess::version(output) ==
        transaction_detail::next_backing_version(spec.base_version);
    if (!version_valid) {
      if (transaction.token) {
        provider.quarantine_unknown(std::move(transaction.token));
      }
      abort_lease(true);
      poison_pipeline = true;
      effective_certainty = VirtualRunWriteCertainty::UnknownMayWrite;
      transaction_detail::clear_transaction(transaction);
      return Status::fail(Reason::DeviceLost);
    }
    if (transaction.scan) {
      if (!transaction.output_lease) {
        if (transaction.token) {
          provider.quarantine_unknown(std::move(transaction.token));
        }
        abort_lease(true);
        poison_pipeline = true;
        effective_certainty = VirtualRunWriteCertainty::UnknownMayWrite;
        transaction_detail::clear_transaction(transaction);
        return Status::fail(Reason::PipelinePoisoned);
      }
      transaction.output_lease.authority->virtual_transactions()
          .commit_virtual_transaction_lease(
              std::move(transaction.output_lease),
              transaction_detail::next_backing_version(spec.base_version));
    }
    VirtualBackingAccess::clear_recovery(output);
    transaction_detail::clear_transaction(transaction);
    return Status::success();
  }
  case VirtualBackingTransactionResult::KnownNoWrite: {
    if (transaction.token) {
      provider.abort_known(std::move(transaction.token));
    }
    abort_lease(false);
    VirtualBackingAccess::clear_recovery(output);
    transaction_detail::clear_transaction(transaction);
    effective_certainty = VirtualRunWriteCertainty::KnownNoWrite;
    return Status::fail(Reason::PipelineInvalid);
  }
  case VirtualBackingTransactionResult::UnknownMayWrite:
    if (transaction.token) {
      provider.quarantine_unknown(std::move(transaction.token));
    }
    abort_lease(true);
    poison_pipeline = true;
    effective_certainty = VirtualRunWriteCertainty::UnknownMayWrite;
    transaction_detail::clear_transaction(transaction);
    return Status::fail(Reason::DeviceLost);
  }
  if (transaction.token) {
    provider.quarantine_unknown(std::move(transaction.token));
  }
  abort_lease(true);
  effective_certainty = VirtualRunWriteCertainty::UnknownMayWrite;
  poison_pipeline = true;
  transaction_detail::clear_transaction(transaction);
  return Status::fail(Reason::PipelineInvalid);
}

} // namespace rund::compute::detail
