#include "token.hpp"

#include "match.hpp"

#include <algorithm>
#include <mutex>
#include <new>
#include <vector>

namespace rund::node::accel::detail {
namespace {

struct PickRegistry final {
  std::mutex mutex{};
  std::vector<std::weak_ptr<PickToken>> tokens{};
};

[[nodiscard]] PickRegistry &Registry() {
  static PickRegistry registry{};
  return registry;
}

[[nodiscard]] auto Position(PickRegistry &registry,
                            const std::shared_ptr<void> &owner) {
  const std::owner_less<void> less{};
  return std::lower_bound(registry.tokens.begin(), registry.tokens.end(), owner,
                          [&less](const std::weak_ptr<PickToken> &candidate,
                                  const std::shared_ptr<void> &key) {
                            return less(candidate, key);
                          });
}

} // namespace

rund::AccelDevice SealPick(rund::AccelDevice raw, const BackendOps &ops) {
  if (!raw.check.ok || raw.owner == nullptr || !raw.caps.ok ||
      raw.api != ops.api) {
    return raw;
  }
  if (!rund::kernel::ComputeStorageAlignmentValid(raw.caps.storage_alignment)) {
    return rund::AccelDevice{
        .check = rund::AccelCheck{false, "compute_adapter_capability_invalid"},
        .api = ops.api,
    };
  }
  try {
    std::shared_ptr<PickToken> token =
        std::make_shared<PickToken>(std::move(raw), ops);
    const std::shared_ptr<void> owner = PublicPickOwner(token);
    PickRegistry &registry = Registry();
    {
      std::lock_guard lock{registry.mutex};
      std::erase_if(registry.tokens,
                    [](const std::weak_ptr<PickToken> &candidate) {
                      return candidate.expired();
                    });
      if (registry.tokens.size() == registry.tokens.max_size()) {
        return rund::AccelDevice{
            .check = rund::AccelCheck{false, "compute_adapter_unavailable"},
            .api = ops.api,
        };
      }
      registry.tokens.insert(Position(registry, owner), token);
    }
    rund::AccelDevice sealed = token->raw;
    sealed.owner = owner;
    return sealed;
  } catch (const std::bad_alloc &) {
    return rund::AccelDevice{
        .check = rund::AccelCheck{false, "compute_adapter_unavailable"},
        .api = ops.api,
    };
  }
}

PickAdmission AdmitPick(const rund::AccelDevice &pick) noexcept {
  if (pick.owner == nullptr) {
    return {};
  }
  std::shared_ptr<PickToken> token{};
  PickRegistry &registry = Registry();
  {
    std::lock_guard lock{registry.mutex};
    const auto found = Position(registry, pick.owner);
    if (found == registry.tokens.end() || !SameOwner(*found, pick.owner)) {
      return {};
    }
    token = found->lock();
  }
  if (token == nullptr || !SameObject(token, pick.owner) ||
      token->ops == nullptr || !SameCheck(pick.check, token->raw.check) ||
      pick.api != token->raw.api || !SameCaps(pick.caps, token->raw.caps) ||
      !SameCpuCaps(pick.cpu_caps, token->raw.cpu_caps) ||
      !SameDispatch(pick.backend, token->raw.backend) ||
      !SameBackendInfo(pick.backend_info, token->raw.backend_info)) {
    return {};
  }
  return PickAdmission{.check = rund::AccelCheck{true, "ok"},
                       .token = std::move(token)};
}

} // namespace rund::node::accel::detail
