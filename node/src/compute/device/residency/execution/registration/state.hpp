#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <utility>

namespace rund::compute::detail::residency {
class DirectRecurrenceOwner;
}

namespace rund::compute::detail::residency::registration_detail {

enum class Policy : std::uint8_t { Release, Retain };
enum class Lifecycle : std::uint8_t {
  Active,
  Admitted,
  Frozen,
  Pending,
  Inflight,
  Released,
  Quarantined,
};

class State final {
public:
  static constexpr std::size_t BindingCapacity = 64u;

  struct Binding final {
    std::uint64_t resource{};
    std::uint64_t bytes{};
    std::uint64_t offset_bytes{};
    std::uint64_t element_bytes{};
    std::uint64_t stride_bytes{};
    std::uint64_t count{};
    std::uint32_t usage{};
    std::uint32_t region_first{};
    std::uint32_t region_count{};
    std::uint8_t tier{};
    std::uint8_t role{};
    std::uint64_t registration{};

    [[nodiscard]] constexpr bool
    operator==(const Binding &) const noexcept = default;
  };

  State(const State &) = delete;
  State &operator=(const State &) = delete;

  [[nodiscard]] std::uint64_t nonce() const noexcept { return nonce_; }
  [[nodiscard]] Policy policy() const noexcept { return policy_; }
  [[nodiscard]] std::size_t binding_count() const noexcept {
    return binding_count_;
  }
  [[nodiscard]] const Binding &binding(const std::size_t index) const noexcept {
    return bindings_[index];
  }
  [[nodiscard]] Lifecycle phase() const noexcept {
    return lifecycle_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::shared_ptr<const void> owner() const noexcept {
    return owner_.lock();
  }
  [[nodiscard]] bool
  owner_matches(const std::shared_ptr<const void> &candidate) const noexcept {
    const std::shared_ptr<const void> sealed = owner_.lock();
    if (sealed == nullptr || candidate == nullptr ||
        sealed.get() != candidate.get()) {
      return false;
    }
    return !sealed.owner_before(candidate) && !candidate.owner_before(sealed);
  }

private:
  friend class ::rund::compute::detail::residency::DirectRecurrenceOwner;

  [[nodiscard]] bool transition(const Lifecycle expected,
                                const Lifecycle next) const noexcept {
    Lifecycle observed = expected;
    return lifecycle_.compare_exchange_strong(
        observed, next, std::memory_order_acq_rel, std::memory_order_acquire);
  }
  void set(const Lifecycle next) const noexcept {
    lifecycle_.store(next, std::memory_order_release);
  }
  friend std::shared_ptr<const State> make(Policy,
                                           const std::shared_ptr<const void> &,
                                           std::span<const Binding>) noexcept;

  State(const std::uint64_t nonce, const Policy policy,
        std::shared_ptr<const void> owner,
        const std::span<const Binding> bindings)
      : nonce_(nonce), policy_(policy), owner_(std::move(owner)),
        binding_count_(bindings.size()) {
    std::copy(bindings.begin(), bindings.end(), bindings_.begin());
  }

  const std::uint64_t nonce_;
  const Policy policy_;
  const std::weak_ptr<const void> owner_;
  const std::size_t binding_count_;
  std::array<Binding, BindingCapacity> bindings_{};
  mutable std::atomic<Lifecycle> lifecycle_{Lifecycle::Active};
};

[[nodiscard]] inline std::shared_ptr<const State>
make(const Policy policy, const std::shared_ptr<const void> &owner,
     const std::span<const State::Binding> bindings) noexcept {
  if (owner == nullptr || bindings.empty() ||
      bindings.size() > State::BindingCapacity) {
    return {};
  }
  static std::atomic<std::uint64_t> next{1u};
  static std::atomic<bool> exhausted{false};
  if (exhausted.load(std::memory_order_acquire)) {
    return {};
  }
  const std::uint64_t nonce = next.fetch_add(1u, std::memory_order_relaxed);
  if (nonce == 0u || nonce == std::numeric_limits<std::uint64_t>::max()) {
    exhausted.store(true, std::memory_order_release);
    return {};
  }
  try {
    return std::shared_ptr<const State>(
        new State(nonce, policy, owner, bindings));
  } catch (const std::bad_alloc &) {
    return {};
  }
}

} // namespace rund::compute::detail::residency::registration_detail
