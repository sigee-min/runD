#include "authority.hpp"

#include "model.hpp"

#include "../../../device/residency/registry/cpu_graph_owner.hpp"

namespace rund::compute::detail::graph_reduce {

bool close_cpu_epoch(residency::Authority &authority, CpuEpochReceipt &receipt,
                     const bool success, const bool invalidate_all,
                     residency::CloseInfo *const info) noexcept {
  if (info != nullptr) {
    *info = residency::CloseInfo{};
  }
  if (!receipt.bound_to(authority)) {
    if (info != nullptr) {
      info->check = residency::CloseInfo::Check::Credential;
    }
    return false;
  }
  const bool closed = authority.cpu_graph().close_cpu_epoch(
      receipt.key(), receipt.token(), receipt.generation(), success,
      invalidate_all, info);
  if (closed) {
    receipt.clear();
  }
  return closed;
}

CpuBindResult bind_cpu_epoch(residency::Authority &authority,
                             CpuEpochPermit &permit, const std::uint64_t token,
                             const std::uint64_t generation,
                             residency::CloseInfo *const info) noexcept {
  if (info != nullptr) {
    *info = residency::CloseInfo{};
  }
  const auto close = [&]() noexcept {
    residency::CloseInfo local{};
    const bool closed = authority.cpu_graph().close_cpu_epoch(
        permit.key(), token, generation, false, true, &local);
    if (info != nullptr) {
      *info = local;
    }
    if (closed) {
      permit.release_closed();
      return CpuBindResult::Closed;
    }
    return CpuBindResult::Retained;
  };
  if (!permit.bound_to(authority)) {
    if (info != nullptr) {
      info->check = residency::CloseInfo::Check::Credential;
      info->token = token;
      info->generation = generation;
    }
    return CpuBindResult::Retained;
  }
  if (token == 0u || generation == 0u) {
    return close();
  }
  permit.prepare(token, generation);
  if (!authority.cpu_graph().confirm_cpu_epoch(permit.key(), token,
                                              generation)) {
    if (info != nullptr) {
      info->check = residency::CloseInfo::Check::Credential;
      info->token = token;
      info->generation = generation;
    }
    return close();
  }
  if (!permit.finalize(token, generation)) {
    return close();
  }
  permit.detach();
  return CpuBindResult::Bound;
}

Status authority_status(const residency::AuthorityResult &result) noexcept {
  switch (result.failure) {
  case residency::AuthorityFailure::Busy:
    return Status::fail(Reason::PipelineBusy);
  case residency::AuthorityFailure::Capacity:
    return Status::fail(Reason::PipelineMemoryBudget);
  case residency::AuthorityFailure::Invalid:
    return Status::fail(Reason::PipelineInvalid);
  case residency::AuthorityFailure::None:
    break;
  }
  return Status::fail(Reason::PipelineInvalid);
}

bool discard_keys(residency::Authority &authority,
                  const std::span<const residency::CacheKey> keys,
                  const residency::FrameRegion region) noexcept {
  const residency::AuthorityResult drain =
      authority.begin_discard(keys, region.first, region.count);
  return drain && authority.discard(drain.lease.token);
}

} // namespace rund::compute::detail::graph_reduce
