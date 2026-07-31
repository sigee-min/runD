#pragma once

#include <rund/net/bytes.hpp>
#include <rund/net/drain.hpp>
#include <rund/net/ready.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>

#include <array>
#include <cstddef>

struct WriteDrainSocketPairCleanup {
  int left = -1;
  int right = -1;

  ~WriteDrainSocketPairCleanup();

  WriteDrainSocketPairCleanup() = default;
  WriteDrainSocketPairCleanup(const WriteDrainSocketPairCleanup &) = delete;
  WriteDrainSocketPairCleanup &
  operator=(const WriteDrainSocketPairCleanup &) = delete;
};

[[nodiscard]] bool
MakeWriteDrainSocketPair(WriteDrainSocketPairCleanup &cleanup);
[[nodiscard]] rund::SessionConfig NetWriteDrainRunSpec() noexcept;

template <std::size_t Count>
[[nodiscard]] std::array<std::byte, Count> MakeWriteDrainPayload() {
  std::array<std::byte, Count> payload{};
  for (std::size_t index = 0; index < payload.size(); ++index) {
    payload[index] = static_cast<std::byte>(index & 0x7fu);
  }
  return payload;
}

[[nodiscard]] int RunWriteDrainSuccessCase();
[[nodiscard]] int RunWriteDrainBoundCase();
[[nodiscard]] int RunWriteDrainCallbackCase();
