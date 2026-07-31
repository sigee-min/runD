#pragma once

#include <rund/net/socket.hpp>
#include <rund/session.hpp>

#include <cstddef>
#include <iostream>

#define FRAMEIO_ASSERT(condition)                                              \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "FRAMEIO_ASSERT failed: " #condition " at " << __FILE__     \
                << ':' << __LINE__ << '\n';                                    \
      return false;                                                            \
    }                                                                          \
  } while (false)

struct FrameIoSocketPair {
  rund::net::Socket left{};
  rund::net::Socket right{};
};

[[nodiscard]] bool OpenFrameIoNonblockingPair(FrameIoSocketPair &pair);
[[nodiscard]] rund::SessionConfig FrameIoRunSpec() noexcept;

[[nodiscard]] bool FrameIoSuccessfulWriteReadPreservesBytes();
[[nodiscard]] bool FrameIoPayloadTooLargeRejectsBeforeWrite();
[[nodiscard]] bool FrameIoVectoredCapacityRejectsDirectly();
[[nodiscard]] bool FrameIoDestinationBufferTooSmallRejectsAfterHeader();
[[nodiscard]] bool FrameIoZeroLengthPayloadSucceedsWithNoPayloadBytes();
[[nodiscard]] bool FrameIoBudgetExhaustionReportsIncomplete();
[[nodiscard]] bool FrameIoZeroWriteBudgetReportsIncomplete();
[[nodiscard]] bool FrameIoZeroReadBudgetReportsIncomplete();
[[nodiscard]] bool FrameIoHeaderOnlyReadBudgetReportsIncomplete();
