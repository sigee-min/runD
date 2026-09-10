#pragma once

#include <rund/compute/cache.hpp>

#include <memory>
#include <type_traits>

namespace rund::compute::graph {
struct Fingerprint;
}
namespace rund::compute::detail {
struct ProgramCacheState;
struct ProgramState;
// Borrowed only for the duration of cached_program; never stored in an entry.
struct ProgramBuilder final {
  void *context;
  Result<std::shared_ptr<ProgramState>> (*invoke)(void *);
};
[[nodiscard]] Result<std::shared_ptr<ProgramState>>
cached_program(const std::shared_ptr<ProgramCacheState> &cache,
               const graph::Fingerprint &fingerprint, ProgramBuilder builder);

// Bind without copying or moving the callable, including temporary and
// move-only captures. The compiled overload finishes before this scope exits.
template <class Builder>
[[nodiscard]] Result<std::shared_ptr<ProgramState>>
cached_program(const std::shared_ptr<ProgramCacheState> &cache,
               const graph::Fingerprint &fingerprint, Builder &&builder) {
  using Callable = std::remove_reference_t<Builder>;
  return cached_program(
      cache, fingerprint,
      ProgramBuilder{
          const_cast<void *>(static_cast<const void *>(std::addressof(builder))),
          [](void *context) { return (*static_cast<Callable *>(context))(); }});
}
} // namespace rund::compute::detail
