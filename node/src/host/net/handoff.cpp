#include <rund/net/handoff.hpp>

namespace rund::net::accept {

Prepared prepare(Result &&accepted, const Options options) noexcept {
  if (!accepted) {
    return Prepared{accepted.code()};
  }
  Prepared result{::rund::ReasonCode::Ok};
  result.socket = std::move(accepted.socket);
  if (!options.nonblocking) {
    return result;
  }

  const NonblockingResult configured = nonblocking(result.socket.view(), true);
  result.nonblocking = configured;
  if (!configured) {
    net::result::Access::set(result, configured.code());
  }
  return result;
}

} // namespace rund::net::accept
