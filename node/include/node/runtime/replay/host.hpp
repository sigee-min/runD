#pragma once

#include <rund/host/event.hpp>
#include <node/runtime/replay/host/archive.hpp>
#include <node/runtime/replay/host/evidence.hpp>
#include <node/runtime/replay/host/payload.hpp>
#include <rund/replay/storage.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace rund::node {

[[nodiscard]] std::vector<std::byte>
EncodeHostReplayEvents(std::span<const ::rund::host::Event> events);

// On failure, clears out; decoded events are valid only when this returns true.
[[nodiscard]] bool DecodeHostReplayEvents(std::span<const std::byte> encoded,
                                          std::vector<::rund::host::Event> &out);

namespace replay_detail {

struct HostReplayFieldDiff {
  bool mismatch = false;
  const char *field = "host.detail";
  std::uint64_t expected = 0u;
  std::uint64_t actual = 0u;
};

struct HostReplayDecodeResult {
  ::rund::replay::Code code = ::rund::replay::Code::HostNotLoaded;
  std::vector<::rund::host::Event> events{};
  std::uint64_t event_hash = 0u;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code);
  }
};

[[nodiscard]] HostReplayDecodeResult
DecodeHostReplayEvents(std::span<const std::byte> encoded);

[[nodiscard]] bool HostReplayEventsEqual(const ::rund::host::Event &expected,
                                         const ::rund::host::Event &actual) noexcept;

void AppendHostReplayWindow(std::vector<::rund::host::Event> &out,
                            const std::vector<::rund::host::Event> &events,
                            std::size_t center, std::size_t context);

[[nodiscard]] bool
FindFirstHostReplayEventMismatch(const std::vector<::rund::host::Event> &expected,
                                 const std::vector<::rund::host::Event> &actual,
                                 std::size_t &index) noexcept;

[[nodiscard]] HostReplayFieldDiff
DiffHostReplayEvidence(const HostReplayEvidence &expected,
                       const HostReplayEvidence &actual);

} // namespace replay_detail

} // namespace rund::node
