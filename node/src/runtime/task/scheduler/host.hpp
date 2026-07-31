#pragma once

#include <rund/host/random/seed.hpp>
#include <rund/replay/code.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rund::host {
struct Event;
} // namespace rund::host

namespace rund::node {

namespace replay_detail::payload {
class Bytes;
struct InputBinding;
struct RawByteSource;
struct MatchResult;
struct ResolveResult;
} // namespace replay_detail::payload

namespace scheduler_host {

enum class ReplayInputMode : std::uint8_t {
  Unavailable,
  Live,
  Record,
  Replay,
  Scenario,
};

struct ReplayInputCapture final {
  std::uint64_t token = 0u;
  ::rund::replay::Code code = ::rund::replay::Code::InputCaptureNotStarted;
  std::span<std::byte> bytes{};

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
};

[[nodiscard]] bool ActiveTask() noexcept;
[[nodiscard]] std::int64_t LogicalTimeNs() noexcept;
[[nodiscard]] ::rund::host::random::RunSeed RandomSeed() noexcept;
[[nodiscard]] bool CapturesIngress() noexcept;
[[nodiscard]] bool Record(::rund::host::Event event) noexcept;
[[nodiscard]] bool
Record(::rund::host::Event event,
       const replay_detail::payload::RawByteSource &source) noexcept;
[[nodiscard]] ReplayInputMode InputMode() noexcept;
[[nodiscard]] ReplayInputCapture
BeginInput(const replay_detail::payload::InputBinding &binding) noexcept;
void FailInput(::rund::replay::Code code) noexcept;
void CancelInput(ReplayInputCapture capture) noexcept;
[[nodiscard]] replay_detail::payload::ResolveResult
RejectInput(ReplayInputCapture capture, ::rund::replay::Code code) noexcept;
[[nodiscard]] replay_detail::payload::ResolveResult
FinishInput(const replay_detail::payload::InputBinding &binding,
            ReplayInputCapture capture, std::size_t byte_count) noexcept;
[[nodiscard]] replay_detail::payload::ResolveResult
ReplayInput(const replay_detail::payload::InputBinding &binding) noexcept;

} // namespace scheduler_host

} // namespace rund::node
