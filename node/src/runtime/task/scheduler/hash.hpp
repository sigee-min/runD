#pragma once

#include <rund/reason.hpp>
#include <rund/task/observation.hpp>
#include <rund/task/operation/kind.hpp>
#include <rund/task/stats/slots.hpp>

#include <cstdint>

namespace rund::node {

inline constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
inline constexpr std::uint64_t kFnvPrime = 1099511628211ull;

enum class TraceDomain : std::uint64_t {
  Operation = 0x6f7065726174696full,
  Observation = 0x6f62736572766174ull,
  Host = 0x686f73746576656eull,
};

static_assert(sizeof(int) == sizeof(std::int32_t));
static_assert(sizeof(short) == sizeof(std::uint16_t));
static_assert(sizeof(ReasonCode) == sizeof(std::uint16_t));
static_assert(sizeof(::rund::task::ObservationKind) == sizeof(std::uint16_t));
static_assert(sizeof(::rund::detail::task::OperationKind) ==
              sizeof(std::uint16_t));

inline constexpr void MixHash(std::uint64_t &hash,
                              const std::uint64_t value) noexcept {
  hash ^= value;
  hash *= kFnvPrime;
}

inline constexpr void MixTrace(::rund::detail::task::StatStorage &stats,
                               const std::uint64_t value) noexcept {
  MixHash(::rund::detail::task::Stat(stats,
                                     ::rund::detail::task::StatSlot::TraceHash),
          value);
}

inline constexpr void BeginTrace(::rund::detail::task::StatStorage &stats,
                                 const TraceDomain domain) noexcept {
  MixTrace(stats, static_cast<std::uint64_t>(domain));
}

inline constexpr void HashOperation(
    ::rund::detail::task::StatStorage &stats, const std::uint64_t sequence,
    const ::rund::detail::task::OperationKind kind, const std::uint64_t task_id,
    const std::uint64_t target_id, const std::uint64_t wait_id,
    const std::uint64_t channel_id, const int fd, const short interest,
    const short revents, const std::int64_t deadline_ns,
    const std::uint64_t value_count, const std::uint64_t match_sequence,
    const std::uint64_t baton_epoch, const std::uint64_t task_op_ordinal,
    const std::uint64_t target_op_ordinal, const std::uint64_t region_id,
    const std::uint64_t epoch_id, const std::uint64_t logical_count,
    const std::uint64_t order_hash, const ReasonCode side_exit_code,
    const ReasonCode code) noexcept {
  BeginTrace(stats, TraceDomain::Operation);
  MixTrace(stats, sequence);
  MixTrace(stats, static_cast<std::uint16_t>(kind));
  MixTrace(stats, task_id);
  MixTrace(stats, target_id);
  MixTrace(stats, wait_id);
  MixTrace(stats, channel_id);
  MixTrace(stats, static_cast<std::uint64_t>(static_cast<std::int64_t>(fd)));
  MixTrace(stats, static_cast<std::uint16_t>(interest));
  MixTrace(stats, static_cast<std::uint16_t>(revents));
  MixTrace(stats, static_cast<std::uint64_t>(deadline_ns));
  MixTrace(stats, value_count);
  MixTrace(stats, match_sequence);
  MixTrace(stats, baton_epoch);
  MixTrace(stats, task_op_ordinal);
  MixTrace(stats, target_op_ordinal);
  MixTrace(stats, region_id);
  MixTrace(stats, epoch_id);
  MixTrace(stats, logical_count);
  MixTrace(stats, order_hash);
  MixTrace(stats, static_cast<std::uint16_t>(side_exit_code));
  MixTrace(stats, static_cast<std::uint16_t>(code));
}

inline constexpr void
HashObservation(::rund::detail::task::StatStorage &stats,
                const ::rund::task::Observation &observation) noexcept {
  BeginTrace(stats, TraceDomain::Observation);
  MixTrace(stats, observation.sequence);
  MixTrace(stats, static_cast<std::uint16_t>(observation.kind));
  MixTrace(stats, observation.task_id);
  MixTrace(stats, observation.wait_id);
  MixTrace(stats, static_cast<std::uint64_t>(
                      static_cast<std::int64_t>(observation.fd)));
  MixTrace(stats, static_cast<std::uint16_t>(observation.interest));
  MixTrace(stats, static_cast<std::uint16_t>(observation.revents));
  MixTrace(stats, static_cast<std::uint64_t>(observation.deadline_ns));
  MixTrace(stats, static_cast<std::uint16_t>(observation.reason_code));
}

inline constexpr void HashHost(::rund::detail::task::StatStorage &stats,
                               const std::uint64_t event_hash) noexcept {
  BeginTrace(stats, TraceDomain::Host);
  MixTrace(stats, event_hash);
}

} // namespace rund::node
