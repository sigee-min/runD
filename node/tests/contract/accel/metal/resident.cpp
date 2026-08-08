#include <accel/api.hpp>
#include <accel/device.hpp>

#include "src/accel/backend/token.hpp"
#include "src/accel/metal/resident.hpp"
#include "src/accel/metal/resident/access.hpp"
#include "src/accel/metal/state.hpp"

#include <node/accel/pick.hpp>

#include <mutex>

namespace node_accel_contract {

bool MetalResidentBufferRegistryValidatesPrivateRefs(
    const rund::AccelDevice &pick) {
  if (!pick.check.ok || !pick.caps.ok || pick.api != rund::AccelApi::Metal) {
    return true;
  }
  const auto token = rund::node::accel::detail::AdmitPick(pick);
  const rund::AccelDevice *const raw = token == nullptr ? nullptr : &token->raw;
  if (raw == nullptr) {
    return false;
  }
  auto *const adapter = static_cast<rund::node::accel::detail::MetalAdapter *>(
      raw->backend.context);
  if (adapter == nullptr) {
    return false;
  }

  const bool validates = [&] {
    const rund::node::accel::detail::ResidentDesc read_desc{
        .bytes = 64u,
        .element_bytes = 4u,
        .stride_bytes = 4u,
        .count = 16u,
        .usage = rund::kernel::kResidentUsageRead,
    };
    const rund::node::accel::detail::MetalResidentBufferResult first =
        rund::node::accel::detail::CreateMetalResidentBuffer(*raw, read_desc);
    const rund::node::accel::detail::MetalResidentBufferResult second =
        rund::node::accel::detail::CreateMetalResidentBuffer(*raw, read_desc);
    if (!first.check.ok || !second.check.ok || first.ref.id == 0u ||
        second.ref.id == 0u || first.ref.id == second.ref.id) {
      return false;
    }

    const rund::node::accel::detail::MetalResidentBufferResult found =
        rund::node::accel::detail::LookupMetalResidentBuffer(*raw, first.ref,
                                                             first.handle);
    if (!found.check.ok || found.ref.id != first.ref.id ||
        found.ref.bytes != first.ref.bytes ||
        found.ref.usage != rund::kernel::kResidentUsageRead) {
      return false;
    }

    rund::kernel::ResidentBufferRef unknown = first.ref;
    unknown.id = second.ref.id + 1000000u;
    if (rund::node::accel::detail::LookupMetalResidentBuffer(*raw, unknown,
                                                             first.handle)
            .check.ok) {
      return false;
    }

    rund::kernel::ResidentBufferRef wrong_usage = first.ref;
    wrong_usage.usage = rund::kernel::kResidentUsageWrite;
    if (rund::node::accel::detail::LookupMetalResidentBuffer(*raw, wrong_usage,
                                                             first.handle)
            .check.ok) {
      return false;
    }

    rund::kernel::ResidentBufferRef too_large = first.ref;
    too_large.bytes = first.ref.bytes + 1u;
    if (rund::node::accel::detail::LookupMetalResidentBuffer(*raw, too_large,
                                                             first.handle)
            .check.ok) {
      return false;
    }

    rund::kernel::ResidentBufferRef too_small_for_extent = first.ref;
    too_small_for_extent.bytes = first.ref.bytes - 1u;
    if (rund::node::accel::detail::LookupMetalResidentBuffer(
            *raw, too_small_for_extent, first.handle)
            .check.ok) {
      return false;
    }

    const rund::AccelDevice other_pick =
        rund::node::accel::PickAccel(rund::AccelPolicy{
            .preferred = {rund::AccelApi::Metal, rund::AccelApi::Auto,
                          rund::AccelApi::Auto},
            .preferred_count = 1u,
        });
    if (!other_pick.check.ok) {
      return false;
    }
    const auto other_token = rund::node::accel::detail::AdmitPick(other_pick);
    const rund::AccelDevice *const other_raw =
        other_token == nullptr ? nullptr : &other_token->raw;
    return other_raw != nullptr &&
           !rund::node::accel::detail::LookupMetalResidentBuffer(
                *other_raw, first.ref, first.handle)
                .check.ok;
  }();
  if (!validates) {
    return false;
  }
  auto &resident = rund::node::accel::detail::MetalResidents(*adapter);
  std::lock_guard lock{resident.mutex};
  return resident.buffers.empty();
}

} // namespace node_accel_contract
