#pragma once

#include <atomic>
#include <cstdint>
#include <rund/net/ready/set.hpp>

namespace rund::node {

enum class ReactorReadySetActivation : std::uint8_t {
  Invalid,
  ReuseGeneration,
  NeedsSlotId,
};

struct ReactorReadySetIdentityState {
  ::rund::net::ready::Set handle{};
  bool live = false;
};

class ReactorReadySetIdentityOwner final {
public:
  explicit ReactorReadySetIdentityOwner(
      std::uint64_t next_slot_id = 1u) noexcept;

  [[nodiscard]] static bool empty(::rund::net::ready::Set handle) noexcept;
  [[nodiscard]] static bool valid(::rund::net::ready::Set handle) noexcept;
  [[nodiscard]] static bool same(::rund::net::ready::Set left,
                                 ::rund::net::ready::Set right) noexcept;
  [[nodiscard]] static bool
  live(const ReactorReadySetIdentityState &state) noexcept;
  [[nodiscard]] static bool matches(const ReactorReadySetIdentityState &state,
                                    ::rund::net::ready::Set handle) noexcept;
  [[nodiscard]] static ReactorReadySetActivation
  activation(const ReactorReadySetIdentityState &state) noexcept;

  [[nodiscard]] bool activate(ReactorReadySetIdentityState &state,
                              ::rund::net::ready::Set *handle) noexcept;
  [[nodiscard]] bool retire(ReactorReadySetIdentityState &state,
                            ::rund::net::ready::Set expected,
                            ::rund::net::ready::Set *tombstone) noexcept;

private:
  [[nodiscard]] bool issue_slot_id(std::uint64_t *slot_id) noexcept;

  std::atomic<std::uint64_t> next_slot_id_;
};

[[nodiscard]] ReactorReadySetIdentityOwner &
ProcessReadySetIdentityOwner() noexcept;

} // namespace rund::node
