#include "local.hpp"

namespace server_inline_detail {

[[nodiscard]] bool HasCounts(const rund::net::server::Result &result,
                             const std::uint32_t completed,
                             const std::uint32_t failed,
                             const std::uint32_t stopped) noexcept {
  return result.accepted == 1u && result.started == 1u &&
         result.completed == completed && result.failed == failed &&
         result.stopped == stopped && result.rejected == 0u;
}

[[nodiscard]] std::uint64_t
CountEvents(const rund::Session::Result &run,
            const rund::host::EventKind kind) noexcept {
  std::uint64_t count = 0u;
  for (const rund::host::Event &event : run.events()) {
    count += event.kind == kind ? 1u : 0u;
  }
  return count;
}

} // namespace server_inline_detail
