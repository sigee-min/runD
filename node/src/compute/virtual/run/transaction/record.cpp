#include "../transaction.hpp"

#include "../../../device/residency/registry/transaction_owner.hpp"

#include <limits>

namespace rund::compute::detail {

Status record_transaction_output(
    VirtualRunTransaction &transaction,
    const residency::VirtualTransactionOwner &owner,
    const std::uint64_t lease_token, const std::uint32_t frame,
    const residency::CacheKey key, const residency::FrameTier tier) noexcept {
  if (!transaction.started || !transaction.scan) {
    return Status::success();
  }
  const std::uint64_t owner_generation = transaction.token.generation();
  if (frame == std::numeric_limits<std::uint32_t>::max() ||
      owner_generation == 0u || !transaction.token) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (!owner.owns(transaction.authority)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (!owner.tag_virtual_transaction_output(lease_token, frame, key, tier,
                                            owner_generation)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const residency::VirtualTransactionLease::Row row{
      .frame = frame,
      .key = key,
      .tier = tier,
      .role = residency::FrameRole::Output,
      .owner_generation = owner_generation};
  if (transaction.journal == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (transaction.journal_count > VirtualRunTransaction::JournalCapacity) {
    return Status::fail(Reason::PipelineInvalid);
  }
  constexpr std::size_t none = std::numeric_limits<std::size_t>::max();
  std::size_t frame_index = none;
  std::size_t key_index = none;
  for (std::size_t index = 0u; index < transaction.journal_count; ++index) {
    const residency::VirtualTransactionLease::Row &current =
        (*transaction.journal)[index];
    if (current.owner_generation == 0u ||
        current.role != residency::FrameRole::Output ||
        (current.tier != residency::FrameTier::Device &&
         current.tier != residency::FrameTier::Host)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (current.frame == frame) {
      if (frame_index != none) {
        return Status::fail(Reason::PipelineInvalid);
      }
      frame_index = index;
    }
    if (current.key == key) {
      if (key_index != none) {
        return Status::fail(Reason::PipelineInvalid);
      }
      key_index = index;
    }
  }
  if (frame_index == none && key_index == none) {
    if (transaction.journal_count >= VirtualRunTransaction::JournalCapacity) {
      return Status::fail(Reason::PipelineCapacity);
    }
    (*transaction.journal)[transaction.journal_count++] = row;
    transaction.rows_active = true;
    return Status::success();
  }
  if (frame_index != none && frame_index == key_index) {
    if ((*transaction.journal)[frame_index].owner_generation !=
        owner_generation) {
      return Status::fail(Reason::PipelineInvalid);
    }
    (*transaction.journal)[frame_index] = row;
    transaction.rows_active = true;
    return Status::success();
  }
  if (frame_index != none && key_index == none) {
    if ((*transaction.journal)[frame_index].owner_generation !=
        owner_generation) {
      return Status::fail(Reason::PipelineInvalid);
    }
    (*transaction.journal)[frame_index] = row;
    transaction.rows_active = true;
    return Status::success();
  }
  if (frame_index == none && key_index != none) {
    const residency::VirtualTransactionLease::Row &current =
        (*transaction.journal)[key_index];
    if (current.owner_generation != owner_generation ||
        current.role != residency::FrameRole::Output ||
        row.role != residency::FrameRole::Output) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (current.tier == residency::FrameTier::Host &&
        tier == residency::FrameTier::Device) {
      return Status::success();
    }
    if (current.tier != residency::FrameTier::Device ||
        tier != residency::FrameTier::Host) {
      return Status::fail(Reason::PipelineInvalid);
    }
    (*transaction.journal)[key_index] = row;
    transaction.rows_active = true;
    return Status::success();
  }
  const residency::VirtualTransactionLease::Row &frame_row =
      (*transaction.journal)[frame_index];
  const residency::VirtualTransactionLease::Row &key_row =
      (*transaction.journal)[key_index];
  if (frame_row.owner_generation != owner_generation ||
      key_row.owner_generation != owner_generation ||
      frame_row.tier != residency::FrameTier::Host ||
      key_row.tier != residency::FrameTier::Device ||
      tier != residency::FrameTier::Host ||
      frame_row.role != residency::FrameRole::Output ||
      key_row.role != residency::FrameRole::Output ||
      row.role != residency::FrameRole::Output) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::size_t last = transaction.journal_count - 1u;
  if (frame_index == last) {
    if (key_index != last) {
      (*transaction.journal)[key_index] = (*transaction.journal)[last];
    }
    --transaction.journal_count;
    (*transaction.journal)[key_index] = row;
  } else {
    (*transaction.journal)[frame_index] = row;
    if (key_index != last) {
      (*transaction.journal)[key_index] = (*transaction.journal)[last];
    }
    --transaction.journal_count;
  }
  transaction.rows_active = true;
  return Status::success();
}

} // namespace rund::compute::detail
