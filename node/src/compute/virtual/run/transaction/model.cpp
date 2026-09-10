#include "internal.hpp"

#include "../../../device/residency/pool.hpp"
#include "../../../pipeline/local.hpp"
#include "../../backing.hpp"
#include "../../state.hpp"

#include <limits>

namespace rund::compute::detail::transaction_detail {

std::uint64_t next_backing_version(const std::uint64_t version) noexcept {
  const std::uint64_t next = version + 1u;
  return next == 0u ? 1u : next;
}

bool transaction_regions(const VirtualRunTransaction &transaction,
                         std::array<residency::FrameRegion, 4u> &regions,
                         std::size_t &count) noexcept {
  count = 0u;
  if (!transaction.scan) {
    return true;
  }
  regions[count++] = transaction.output_regions[0u];
  regions[count++] = transaction.output_regions[1u];
  regions[count++] = transaction.host_output_regions[0u];
  regions[count++] = transaction.host_output_regions[1u];
  return true;
}

bool valid_captured_region(const residency::FrameRegion region,
                           const residency::FrameTier tier) noexcept {
  return region.tier == tier && region.role == residency::FrameRole::Output &&
         region.count != 0u &&
         static_cast<std::uint64_t>(region.first) + region.count <=
             std::numeric_limits<std::uint32_t>::max();
}

bool disjoint_captured_regions(
    const std::array<residency::FrameRegion, 2u> &regions) noexcept {
  const std::uint64_t first_end =
      static_cast<std::uint64_t>(regions[0u].first) + regions[0u].count;
  const std::uint64_t second_end =
      static_cast<std::uint64_t>(regions[1u].first) + regions[1u].count;
  return static_cast<std::uint64_t>(regions[0u].first) >= second_end ||
         static_cast<std::uint64_t>(regions[1u].first) >= first_end;
}

residency::Authority *
scan_authority(const VirtualPipelineState &state) noexcept {
  if (state.pipeline == nullptr || state.alternate_pipeline == nullptr ||
      state.pipeline == state.alternate_pipeline ||
      state.pipeline->residency_pool == nullptr ||
      state.alternate_pipeline->residency_pool == nullptr) {
    return nullptr;
  }
  residency::Pool *const primary = state.pipeline->residency_pool.get();
  residency::Pool *const alternate =
      state.alternate_pipeline->residency_pool.get();
  if (primary != alternate || primary->registry == nullptr ||
      alternate->registry == nullptr) {
    return nullptr;
  }
  residency::Authority *const authority = &primary->authority();
  return authority == &alternate->authority() ? authority : nullptr;
}

bool valid_output_lease(const VirtualRunTransaction &transaction,
                        const residency::Authority *const authority) noexcept {
  const residency::VirtualTransactionLease &lease = transaction.output_lease;
  if (lease.authority == nullptr) {
    return !lease.active && !lease.gate.owns_lock();
  }
  return lease.authority == authority && lease.active && lease.gate.owns_lock();
}

Status
validate_transaction_base(const VirtualPipelineState &state,
                          const VirtualRunTransaction &transaction) noexcept {
  if (!transaction.started || transaction.output == nullptr ||
      transaction.provider == nullptr || !transaction.token) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (transaction.scan) {
    residency::Authority *const authority = scan_authority(state);
    if (authority == nullptr || transaction.authority != authority ||
        !valid_output_lease(transaction, authority)) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  const VirtualBacking &output = *transaction.output;
  const VirtualBackingTransactionSpec spec = transaction.token.spec();
  if (spec.backing_id == 0u || spec.base_version == 0u ||
      spec.logical_bytes == 0u || spec.backing_page_bytes == 0u ||
      spec.backing_page_count == 0u || spec.write_page_bytes == 0u ||
      spec.write_page_count == 0u ||
      spec.logical_bytes != output.size_bytes() ||
      spec.backing_id != VirtualBackingAccess::id(output) ||
      spec.base_version != VirtualBackingAccess::version(output) ||
      spec.write_page_count != transaction.page_count ||
      dynamic_cast<VirtualBackingTransaction *>(transaction.output) !=
          transaction.provider ||
      VirtualBackingAccess::recovery_bytes(output) < spec.logical_bytes) {
    return Status::fail(Reason::PipelineInvalid);
  }
  return Status::success();
}

void clear_transaction(VirtualRunTransaction &transaction) noexcept {
  if (transaction.output != nullptr && !transaction.scan) {
    VirtualBackingAccess::clear_transaction(*transaction.output);
  }
  transaction = VirtualRunTransaction{};
}

} // namespace rund::compute::detail::transaction_detail
