#include "catalog.hpp"

#include "ops/table.hpp"
#include "token.hpp"

#include <iterator>
#include <utility>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] rund::AccelDevice Reject(const rund::AccelApi api,
                                       const char *const reason) {
  return rund::AccelDevice{
      .check = rund::AccelCheck{false, reason},
      .api = api,
  };
}

[[nodiscard]] rund::AccelDevice Admit(const BackendEntry &entry,
                                      const bool allow_fake) {
  if (entry.pick == nullptr || entry.ops == nullptr ||
      entry.api != entry.ops->api || entry.api == rund::AccelApi::Auto) {
    return Reject(rund::AccelApi::Auto, "compute_policy_invalid");
  }
  rund::AccelDevice raw = entry.pick(allow_fake);
  return raw.check.ok ? SealPick(std::move(raw), *entry.ops) : raw;
}

[[nodiscard]] const BackendEntry *
Find(const std::span<const BackendEntry> catalog,
     const rund::AccelApi api) noexcept {
  for (const BackendEntry &entry : catalog) {
    if (entry.api == api) {
      return &entry;
    }
  }
  return nullptr;
}

[[nodiscard]] rund::AccelDevice
PickAuto(const std::span<const BackendEntry> catalog, const bool allow_fake) {
  for (const BackendEntry &entry : catalog) {
    if (!entry.automatic ||
        (entry.api == rund::AccelApi::Fake && !allow_fake)) {
      continue;
    }
    rund::AccelDevice pick = Admit(entry, allow_fake);
    if (pick.check.ok) {
      return pick;
    }
  }
  return Reject(rund::AccelApi::Auto, "compute_adapter_unavailable");
}

} // namespace

rund::AccelDevice PickFromCatalog(const std::span<const BackendEntry> catalog,
                                  const rund::AccelPolicy &policy) {
  if (policy.preferred_count == 0u) {
    return Reject(rund::AccelApi::Auto, "compute_policy_empty");
  }
  if (policy.preferred_count > std::size(policy.preferred) || catalog.empty()) {
    return Reject(rund::AccelApi::Auto, "compute_policy_invalid");
  }

  rund::AccelDevice last =
      Reject(rund::AccelApi::Auto, "compute_adapter_unavailable");
  for (std::uint32_t index = 0u; index < policy.preferred_count; ++index) {
    const rund::AccelApi api = policy.preferred[index];
    if (api == rund::AccelApi::Auto) {
      last = PickAuto(catalog, policy.allow_fake);
    } else {
      const BackendEntry *const entry = Find(catalog, api);
      last = entry == nullptr
                 ? Reject(rund::AccelApi::Auto, "compute_policy_invalid")
                 : Admit(*entry, policy.allow_fake);
    }
    if (last.check.ok) {
      return last;
    }
  }
  return last;
}

} // namespace rund::node::accel::detail
