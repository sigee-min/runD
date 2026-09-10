#pragma once

#include "../cache_model.hpp"

#include <cstdint>
#include <limits>

namespace rund::compute::detail::residency {

enum class AuthorityFailure : std::uint8_t {
  None,
  Invalid,
  Busy,
  Capacity,
};

enum class RegistrationResult : std::uint8_t {
  Done,
  Busy,
  Invalid,
  Quarantined,
};

// Fixed diagnostic snapshot for an Authority close rejection. It carries
// observation only; cleanup and quarantine remain owned by Authority.
struct CloseInfo final {
  static constexpr std::uint32_t NoIndex =
      std::numeric_limits<std::uint32_t>::max();

  enum class Check : std::uint8_t {
    None,
    Credential,
    Bounds,
    Key,
    Invalidate,
    RetireAccess,
    RetireDomain,
    RetirePayload,
    DuplicateFrame,
    Alias,
    Undo,
    Relocation,
    State,
    Persist,
  };

  enum Flag : std::uint8_t {
    Success = 1u << 0u,
    Invalidate = 1u << 1u,
    Fetch = 1u << 2u,
    Relocated = 1u << 3u,
    Retire = 1u << 4u,
  };

  Check check{Check::None};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint32_t binding_index{NoIndex};
  CacheKey binding_key{};
  Access binding_access{Access::Read};
  DirtyExtent binding_prior_dirty{};
  std::uint64_t binding_next_use{};
  std::uint64_t binding_retain_until{NeverUse};
  std::uint8_t flags{};
  bool binding_fetch{};
  bool binding_relocated{};
  bool binding_retire{};
  std::uint32_t frame_index{NoIndex};
  CacheKey frame_key{};
  bool frame_assigned{};
  FrameState frame_state{FrameState::Empty};
  FrameTier frame_tier{FrameTier::Device};
  FrameRole frame_role{FrameRole::Input};
  DirtyExtent binding_dirty{};
  DirtyExtent frame_dirty{};
  std::uint64_t frame_next_use{};
  std::uint64_t frame_retain_until{NeverUse};
  std::uint64_t extent{};
  std::uint64_t view{};
  std::uint32_t claims{};
};

struct ViewActivationResult final {
  AuthorityFailure failure{AuthorityFailure::Invalid};
  std::uint64_t eviction_count{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return failure == AuthorityFailure::None;
  }
};

} // namespace rund::compute::detail::residency
