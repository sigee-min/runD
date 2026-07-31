#pragma once

#include <rund/net/flow/state.hpp>
#include <rund/net/result.hpp>

#include <cstdint>

namespace rund::net::flow {

struct Result : net::Status {
  using Status::Status;

  State state{};
  std::uint64_t requested_bytes = 0u;
};

} // namespace rund::net::flow
