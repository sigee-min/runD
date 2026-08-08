#include <accel/api.hpp>

#include "admission/reference.hpp"
#include "capability.hpp"

#include "../backend/match.hpp"

#include <atomic>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace rund::node::accel::detail {
namespace {

struct ContextTokenDelete final {
  ContextToken *token = nullptr;

  void operator()(ContextToken *const value) const noexcept { delete value; }
};

struct AccelBufferTokenDelete final {
  AccelBufferToken *token = nullptr;

  void operator()(AccelBufferToken *const value) const noexcept {
    delete value;
  }
};

[[nodiscard]] std::uint64_t NextContextId() noexcept {
  static std::atomic<std::uint64_t> next{1u};
  std::uint64_t current = next.load(std::memory_order_relaxed);
  while (current != std::numeric_limits<std::uint64_t>::max()) {
    if (next.compare_exchange_weak(current, current + 1u,
                                   std::memory_order_relaxed,
                                   std::memory_order_relaxed)) {
      return current;
    }
  }
  return 0u;
}

} // namespace

std::shared_ptr<void>
PublicTokenOwner(const std::shared_ptr<ContextToken> &token) noexcept {
  return std::static_pointer_cast<void>(token);
}

std::shared_ptr<ContextToken>
MintContextToken(const rund::AccelApi api,
                 const rund::kernel::ComputeCaps &caps,
                 std::shared_ptr<PickToken> pick) {
  if (pick == nullptr || pick->ops == nullptr || pick->raw.api != api ||
      !SameCaps(pick->raw.caps, caps)) {
    return {};
  }
  const std::uint64_t id = NextContextId();
  if (id == 0u) {
    return {};
  }
  ContextToken *const raw =
      new (std::nothrow) ContextToken(id, std::move(pick));
  if (raw == nullptr) {
    return {};
  }
  try {
    std::unique_ptr<ContextToken, ContextTokenDelete> owner{
        raw, ContextTokenDelete{.token = raw}};
    return std::shared_ptr<ContextToken>{std::move(owner)};
  } catch (...) {
    return {};
  }
}

std::shared_ptr<ContextToken>
LookupContextToken(const std::shared_ptr<void> &owner) {
  if (owner == nullptr) {
    return {};
  }
  const auto *const capability = std::get_deleter<ContextTokenDelete>(owner);
  if (capability == nullptr || capability->token == nullptr) {
    return {};
  }
  return std::shared_ptr<ContextToken>{owner, capability->token};
}

std::shared_ptr<AccelBufferToken>
MakeAccelBufferToken(const std::shared_ptr<ContextToken> &context,
                     rund::Buffer backend,
                     const rund::kernel::ResidentBufferRef &resident,
                     const std::uint64_t byte_extent,
                     const rund::AccelBufferDesc &desc) noexcept {
  if (context == nullptr || backend.handle == nullptr) {
    return {};
  }
  AccelBufferToken *const raw = new (std::nothrow) AccelBufferToken{
      context, std::move(backend), resident, byte_extent, desc};
  if (raw == nullptr) {
    return {};
  }
  try {
    std::unique_ptr<AccelBufferToken, AccelBufferTokenDelete> owner{
        raw, AccelBufferTokenDelete{.token = raw}};
    return std::shared_ptr<AccelBufferToken>{std::move(owner)};
  } catch (...) {
    return {};
  }
}

std::shared_ptr<void>
PublicBufferTokenOwner(
    const std::shared_ptr<AccelBufferToken> &token) noexcept {
  return std::static_pointer_cast<void>(token);
}

std::shared_ptr<AccelBufferToken>
LookupAccelBufferToken(const std::shared_ptr<void> &owner) noexcept {
  if (owner == nullptr) {
    return {};
  }
  const auto *const capability =
      std::get_deleter<AccelBufferTokenDelete>(owner);
  if (capability == nullptr || capability->token == nullptr) {
    return {};
  }
  return std::shared_ptr<AccelBufferToken>{owner, capability->token};
}

bool AccelBufferTokenMatches(
    const ContextAdmission &admission, const rund::AccelBuffer &buffer,
    const std::shared_ptr<AccelBufferToken> &token) noexcept {
  if (!admission.check.ok || token == nullptr || token->context == nullptr ||
      !SameObject(admission.owner, token->context) ||
      !SameObject(buffer.handle, token) ||
      !SameObject(buffer.buffer.handle, token) || !buffer.check.ok ||
      !SameReason(buffer.reason, "ok") ||
      buffer.context_id != admission.context_id ||
      !SameObject(buffer.owner, admission.owner) ||
      !SameCheck(buffer.buffer.check, token->backend_check) ||
      buffer.buffer.id != token->backend_id ||
      buffer.buffer.bytes != token->backend_bytes ||
      buffer.buffer.element_bytes != token->backend_element_bytes ||
      buffer.buffer.stride_bytes != token->backend_stride_bytes ||
      buffer.buffer.count != token->backend_count ||
      buffer.buffer.storage_bytes != token->backend_storage_bytes ||
      buffer.buffer.storage_reused != token->backend_storage_reused ||
      buffer.buffer.usage != token->backend_usage ||
      !SameObject(buffer.buffer.owner, admission.owner) ||
      !SameResidentRef(buffer.resident, token->resident) ||
      buffer.byte_extent != token->byte_extent ||
      buffer.scalar_width_bytes != token->scalar_width_bytes ||
      buffer.count != token->count || buffer.usage != token->usage) {
    return false;
  }
  return true;
}

} // namespace rund::node::accel::detail
