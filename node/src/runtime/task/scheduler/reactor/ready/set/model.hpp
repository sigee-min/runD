#pragma once

#include <cstdint>
#include <rund/net/socket.hpp>
#include <vector>

#include "../../../../../reactor/readiness/state.hpp"

namespace rund::node {

struct ReactorReadySetMember {
  ::rund::net::SocketView socket{};
  ReactorHandle fd = kInvalidReactorHandle;
  std::uint32_t index = 0u;
  ReactorInterest interest = ReactorInterest::None;
};

struct ReactorReadySet {
  std::uint64_t id = 0u;
  std::uint64_t generation = 0u;
  std::vector<ReactorReadySetMember> members{};
  std::uint32_t max_members = 0u;
  std::uint32_t next_member_index = 0u;
  bool live = false;
};

}  // namespace rund::node
