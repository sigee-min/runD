#pragma once

#include <node/runtime/replay/host/archive.hpp>
#include <rund/session.hpp>

#include <utility>

namespace rund::node {
struct ScopeEvidence;
}

namespace rund::detail::session {

struct ResultAccess final {
  [[nodiscard]] static ::rund::Session::Result
  make(const ReasonCode code, const std::uint64_t scope,
       ::rund::node::ScopeEvidence evidence, ::rund::Trace trace,
       ::rund::telemetry::Detail detail = {},
       const std::uint8_t telemetry_clock_reads = 0u);

  [[nodiscard]] static ::rund::Session::Result fail(const ReasonCode code,
                                                    ::rund::Trace trace = {});

  static void finish(::rund::Session::Result &result, const ReasonCode code,
                     ::rund::Trace trace) noexcept {
    result.code_ = code;
    result.trace_ = std::move(trace);
  }

  static void timing(::rund::Session::Result &result,
                     const ::rund::telemetry::Detail detail,
                     const std::uint8_t clock_reads) noexcept {
    result.detail_ = detail;
    result.telemetry_clock_reads_ = clock_reads;
  }

  [[nodiscard]] static const ::rund::telemetry::Detail &
  timing(const ::rund::Session::Result &result) noexcept {
    return result.detail_;
  }

  [[nodiscard]] static std::uint8_t
  telemetry_clock_reads(const ::rund::Session::Result &result) noexcept {
    return result.telemetry_clock_reads_;
  }

  [[nodiscard]] static std::uint64_t
  input_rows(const ::rund::Session::Result &result) noexcept {
    return result.input_rows_;
  }

  [[nodiscard]] static std::uint64_t
  input_bytes(const ::rund::Session::Result &result) noexcept {
    return result.input_bytes_;
  }

  [[nodiscard]] static std::uint64_t
  ready_capacity(const ::rund::Session::Result &result) noexcept {
    return result.ready_capacity_;
  }

  [[nodiscard]] static const ::rund::task::Stats &
  tasks(const ::rund::Session::Result &result) noexcept {
    return result.tasks_;
  }

  [[nodiscard]] static std::vector<::rund::task::Observation>
  take_observations(::rund::Session::Result &result) noexcept {
    return std::move(result.observations_);
  }

  [[nodiscard]] static std::vector<::rund::host::Event>
  take_events(::rund::Session::Result &result) noexcept {
    return std::move(result.events_);
  }

  [[nodiscard]] static ::rund::node::replay_detail::payload::Archive
  take_payloads(::rund::Session::Result &result) noexcept {
    return std::move(
        *static_cast<::rund::node::replay_detail::payload::Archive *>(
            result.storage()));
  }

  [[nodiscard]] static ::rund::Trace
  take_trace(::rund::Session::Result &result) noexcept {
    return std::move(result.trace_);
  }
};

} // namespace rund::detail::session
