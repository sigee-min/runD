#pragma once

#include "cache_model.hpp"

#include <cstdint>

namespace rund::compute::detail::residency {

struct CacheBinding final {
  CacheKey key{};
  std::uint32_t frame{};
  Access access{Access::Read};
  DirtyExtent dirty{};
  DirtyExtent prior_dirty{};
  std::uint64_t next_use{};
  std::uint64_t retain_until{NeverUse};
  bool fetch{};
  // The requested key was physically copied from another frame in the same
  // compatible arena before activate(). Failure invalidates every frame in
  // that copy dependency, because metadata rollback cannot restore bytes.
  bool relocated{};
  // Generic Graph last-consumer reads retire a Transient in the same atomic
  // success commit that publishes the stage outputs. No later drain token may
  // observe a half-retired epoch.
  bool retire_on_success{};
};

struct CacheTransition final {
  CacheKey key{};
  std::uint32_t frame{};
  TransitionKind kind{TransitionKind::Fetch};
  DirtyExtent dirty{};
};
} // namespace rund::compute::detail::residency
