#include "local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/async.hpp>
#include <rund/session.hpp>

#include <cstdint>

namespace package_compute::graph_services {

int CheckSessionCompile() {
  rund::Session session{};
  rund::SessionConfig config{};
  config.workers = 2u;
  config.compile = {.workers = 1u, .capacity = 2u};
  const auto opened = session.open(config);
  if (!opened) {
    return opened.exit_code();
  }
  auto device = rund::compute::open(session, rund::compute::Target::cpu());
  const int operation = [&]() -> int {
    if (!device) {
      return device.exit_code();
    }
    if (device->compile().workers != 1u || device->compile().capacity != 2u) {
      return 2;
    }
    auto pending = rund::compute::on(*device)
                       .map<std::int32_t>("session-package-async", 4u,
                                          [](auto value) { return value + 1; })
                       .compile_async();
    if (!pending) {
      return pending.exit_code();
    }
    const auto compiled = pending->get();
    return compiled ? 0 : compiled.exit_code();
  }();
  const auto closed = session.close();
  if (operation != 0 && operation != 2) {
    return operation;
  }
  if (!closed) {
    return closed.exit_code();
  }
  if (operation != 0) {
    return operation;
  }
  auto stopped = rund::compute::on(*device)
                     .map<std::int32_t>("session-package-stopped", 4u,
                                        [](auto value) { return value + 1; })
                     .compile_async();
  return !stopped && stopped.reason() ==
                         rund::compute::Reason::AsyncCompileUnavailable
             ? 0
             : 2;
}

} // namespace package_compute::graph_services
