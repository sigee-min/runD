#pragma once

#if !defined(RUND_NODE_OPEN_PROBE)
#error "compute open probe is available only to Node contract builds"
#endif

#include <cstdint>

namespace rund::compute::detail {

class ScopedOpenProbe final {
public:
  explicit ScopedOpenProbe(std::uint64_t &count) noexcept;
  ~ScopedOpenProbe();

  ScopedOpenProbe(const ScopedOpenProbe &) = delete;
  ScopedOpenProbe &operator=(const ScopedOpenProbe &) = delete;
  ScopedOpenProbe(ScopedOpenProbe &&) = delete;
  ScopedOpenProbe &operator=(ScopedOpenProbe &&) = delete;

private:
  std::uint64_t *previous_{};
};

void observe_open_config() noexcept;

} // namespace rund::compute::detail
