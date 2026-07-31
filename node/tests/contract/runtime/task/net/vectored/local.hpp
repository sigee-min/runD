#pragma once

#include <rund/net/socket.hpp>
#include <rund/session.hpp>

#include <array>
#include <cstddef>
#include <iostream>
#include <string>

#define VECTORED_ASSERT(condition)                                             \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "VECTORED_ASSERT failed: " #condition << " at " << __FILE__ \
                << ':' << __LINE__ << '\n';                                    \
      return false;                                                            \
    }                                                                          \
  } while (false)

struct SocketPair {
  rund::net::Socket left{};
  rund::net::Socket right{};
};

[[nodiscard]] bool OpenSocketPair(SocketPair &pair);
[[nodiscard]] std::string BytesToString(const std::array<std::byte, 6u> &bytes);

[[nodiscard]] bool NetVectoredSendRecvPreservesSliceOrder();
[[nodiscard]] bool NetVectoredPartialCompletionHashesCompletedPrefix();
[[nodiscard]] bool NetVectoredRejectsInvalidSlicesAndCapacity();
[[nodiscard]] bool NetVectoredRejectsImpossibleSliceMetadata();
