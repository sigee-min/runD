#pragma once

#include "../local.hpp"

#include <array>
#include <cstddef>
#include <rund/net/bytes.hpp>
#include <rund/net/socket.hpp>

struct BasicEventCase {
  bool socketpair_ok = false;
  rund::Session::Result report{};
  rund::net::Socket left{};
  rund::net::Socket right{};
  std::array<std::byte, 2u> out{std::byte{'n'}, std::byte{'b'}};
  std::array<std::byte, 2u> in{};
  std::array<std::byte, 1u> zero_out{std::byte{'0'}};
  std::array<std::byte, 1u> zero_in{};
  rund::net::SendResult send{};
  rund::net::ReceiveResult recv{};
  rund::net::SendResult zero_send{};
  rund::net::ReceiveResult zero_recv{};
};

[[nodiscard]] BasicEventCase RunBasicEventScenario();
[[nodiscard]] bool
BasicEventTransferEvidenceMatches(const BasicEventCase &event);
[[nodiscard]] bool BasicEventZeroEvidenceMatches(const BasicEventCase &event);
