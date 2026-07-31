#pragma once

#include <rund/net/bytes.hpp>
#include <rund/net/ready.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

namespace rund::net::drain {

struct Budget {
  std::uint32_t max_operations = 64u;
};

struct ReadResult : net::Status {
  using Status::Status;

  std::uint64_t reads = 0u;
  std::uint64_t bytes = 0u;
  bool would_block = false;
  bool budget_exhausted = false;
  bool handler_stopped = false;
};

struct WriteResult : net::Status {
  using Status::Status;

  std::uint64_t writes = 0u;
  std::uint64_t bytes = 0u;
  bool all_written = false;
  bool would_block = false;
  bool budget_exhausted = false;
  bool handler_stopped = false;
};

namespace detail {

using ReadHandler = bool (*)(const void *, std::span<const std::byte>) noexcept;
using WriteHandler = bool (*)(const void *, std::uint64_t, SendResult) noexcept;

[[nodiscard]] ReadResult read(ready::Ticket &&ticket,
                              std::span<std::byte> buffer, Budget budget,
                              const void *state, ReadHandler handler) noexcept;

[[nodiscard]] WriteResult write(ready::Ticket &&ticket,
                                std::span<const std::byte> bytes, Budget budget,
                                const void *state,
                                WriteHandler handler) noexcept;

[[nodiscard]] inline ReadResult
fail_read(const ::rund::ReasonCode code) noexcept {
  return ReadResult{code};
}

[[nodiscard]] inline WriteResult
fail_write(const ::rund::ReasonCode code) noexcept {
  return WriteResult{code};
}

} // namespace detail

template <typename Callback>
[[nodiscard]] ReadResult
read(ready::Ticket &&ticket, const std::span<std::byte> buffer,
     const Budget budget, Callback &&callback) noexcept {
  static_assert(
      std::is_invocable_r_v<bool, Callback &, std::span<const std::byte>>,
      "drain::read callback must return bool");
  const auto invoke = [](const void *const state,
                         const std::span<const std::byte> bytes) noexcept {
    using State = std::remove_reference_t<Callback>;
    return std::invoke(*const_cast<State *>(static_cast<const State *>(state)),
                       bytes);
  };
  return detail::read(std::move(ticket), buffer, budget,
                      std::addressof(callback), invoke);
}

template <typename Callback>
[[nodiscard]] WriteResult
write(ready::Ticket &&ticket, std::span<const std::byte> bytes,
      const Budget budget, Callback &&callback) noexcept {
  static_assert(
      std::is_invocable_r_v<bool, Callback &, std::uint64_t, SendResult>,
      "drain::write callback must return bool");
  const auto invoke = [](const void *const state, const std::uint64_t completed,
                         const SendResult sent) noexcept {
    using State = std::remove_reference_t<Callback>;
    return std::invoke(*const_cast<State *>(static_cast<const State *>(state)),
                       completed, sent);
  };
  return detail::write(std::move(ticket), bytes, budget,
                       std::addressof(callback), invoke);
}

[[nodiscard]] inline WriteResult write(ready::Ticket &&ticket,
                                       const std::span<const std::byte> bytes,
                                       const Budget budget) noexcept {
  const auto keep_writing = [](const std::uint64_t, const SendResult) noexcept {
    return true;
  };
  return write(std::move(ticket), bytes, budget, keep_writing);
}

[[nodiscard]] inline WriteResult
write(ready::Ticket &&ticket, const std::span<const std::byte> bytes) noexcept {
  return write(std::move(ticket), bytes, Budget{});
}

} // namespace rund::net::drain
