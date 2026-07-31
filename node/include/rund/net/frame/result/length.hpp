#pragma once

#include <rund/net/result.hpp>

#include <cstdint>
#include <string_view>

namespace rund::net::frame {

struct Result : net::Status {
  using net::Status::Status;

  std::uint32_t bytes = 0u;
};

} // namespace rund::net::frame
