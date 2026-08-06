#pragma once

#include "access.hpp"
#include "socket.hpp"

#include "../../../runtime/platform/io.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <unordered_map>

namespace rund::net {

struct SocketSlot;
using SocketIndex = std::unordered_map<int, SocketSlot *>;

struct SocketHot final {
  std::atomic<int> native{-1};
  std::atomic<std::uint64_t> generation{0u};
  std::atomic<std::uint32_t> readers{0u};
  bool closing = false;
};

struct SocketSlot {
  SocketHot hot{};
  node::NativeFdIdentity identity{};
  SocketRegistryOwner active_owner{};
  SocketIndex::node_type index{};
  SocketSlot *next = nullptr;
  std::unique_ptr<SocketSlot> storage{};
};

static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
static_assert(std::atomic<int>::is_always_lock_free);
static_assert(std::atomic<std::uint32_t>::is_always_lock_free);
static_assert(std::is_standard_layout_v<SocketHot>);
static_assert(sizeof(SocketHot) <= 32u);
static_assert(sizeof(void *) != 8u || sizeof(SocketSlot) <= 128u);
static_assert(sizeof(void *) != 8u || sizeof(SocketRegistryOwner) == 24u);

struct SocketRegistryStats final {
  std::size_t slots{};
  std::size_t live{};
  std::size_t reusable{};
  std::size_t burned{};
};

class SocketRegistry final {
public:
  SocketRegistry() noexcept = default;
  ~SocketRegistry() noexcept;
  SocketRegistry(const SocketRegistry &) = delete;
  SocketRegistry &operator=(const SocketRegistry &) = delete;

  [[nodiscard]] SocketSlot *find(int native) noexcept;
  [[nodiscard]] const SocketSlot *find(int native) const noexcept;

  // Admission must own one capacity reservation before calling bind(). The
  // returned slot address remains valid for the process lifetime.
  [[nodiscard]] SocketSlot *bind(int native,
                                 const node::NativeFdIdentity &identity);
  void release(SocketSlot &slot) noexcept;

  [[nodiscard]] SocketRegistryStats stats() const noexcept;

private:
  SocketIndex live_{};
  std::unique_ptr<SocketSlot> slots_{};
  SocketSlot *free_ = nullptr;
  std::size_t slot_count_ = 0u;
  std::size_t reusable_ = 0u;
  std::size_t burned_ = 0u;
};

namespace registry {

inline constexpr std::uint64_t exhausted =
    std::numeric_limits<std::uint64_t>::max();

[[nodiscard]] bool active(std::uint64_t generation) noexcept;
[[nodiscard]] std::uint64_t load(const SocketSlot &slot) noexcept;
[[nodiscard]] std::uint64_t activate(SocketSlot &slot) noexcept;
void retire(SocketSlot &slot) noexcept;
[[nodiscard]] bool retire(SocketSlot &slot, std::uint64_t generation) noexcept;
void wait(const SocketSlot &slot) noexcept;

} // namespace registry

[[nodiscard]] std::shared_mutex &RegistryGate() noexcept;
[[nodiscard]] SocketRegistry &Registry() noexcept;

[[nodiscard]] bool SameIdentity(const node::NativeFdIdentity &left,
                                const node::NativeFdIdentity &right) noexcept;
[[nodiscard]] Socket MakeAdmittedSocket(SocketSlot &slot,
                                        std::uint64_t generation) noexcept;

[[nodiscard]] bool HasOwner(const SocketRegistryOwner &owner) noexcept;
[[nodiscard]] bool SameOwner(const SocketSlot &slot,
                             const SocketRegistryOwner &owner) noexcept;
void AssignOwner(SocketSlot &slot, const SocketRegistryOwner &owner) noexcept;
[[nodiscard]] SocketRegistryOwner TakeOwner(SocketSlot &slot) noexcept;
void ReleaseRuntimeRegistryOwner(const SocketRegistryOwner &owner) noexcept;
[[nodiscard]] bool
ReserveRuntimeRegistryOwner(const SocketRegistryOwner &owner) noexcept;

[[nodiscard]] SocketAdmission AdmitNativeSocketImpl(int native_socket) noexcept;

} // namespace rund::net
