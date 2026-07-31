#pragma once

#include <rund/evidence/numeric/contract.hpp>

#include <cstdint>

namespace rund::cluster {

struct NodeId {
  std::uint64_t value = 0u;
  [[nodiscard]] constexpr explicit operator bool() const { return value != 0u; }
  friend constexpr bool operator==(const NodeId &, const NodeId &) = default;
};

struct JobId {
  std::uint64_t value = 0u;
  [[nodiscard]] constexpr explicit operator bool() const { return value != 0u; }
  friend constexpr bool operator==(const JobId &, const JobId &) = default;
};

struct ShardId {
  std::uint64_t value = 0u;
  [[nodiscard]] constexpr explicit operator bool() const { return value != 0u; }
  friend constexpr bool operator==(const ShardId &, const ShardId &) = default;
};

struct WorkId {
  std::uint64_t value = 0u;
  [[nodiscard]] constexpr explicit operator bool() const { return value != 0u; }
  friend constexpr bool operator==(const WorkId &, const WorkId &) = default;
};

struct InputVersion {
  std::uint64_t value = 0u;
  [[nodiscard]] constexpr explicit operator bool() const { return value != 0u; }
  friend constexpr bool operator==(const InputVersion &,
                                   const InputVersion &) = default;
};

struct LogicalTime {
  std::uint64_t value = 0u;
  friend constexpr bool operator==(const LogicalTime &,
                                   const LogicalTime &) = default;
};

struct CheckpointId {
  std::uint64_t value = 0u;
  friend constexpr bool operator==(const CheckpointId &,
                                   const CheckpointId &) = default;
};

struct ProgramId {
  std::uint64_t value = 0u;
  [[nodiscard]] constexpr explicit operator bool() const { return value != 0u; }
  friend constexpr bool operator==(const ProgramId &,
                                   const ProgramId &) = default;
};

struct FoldId {
  std::uint64_t value = 0u;
  [[nodiscard]] constexpr explicit operator bool() const { return value != 0u; }
  friend constexpr bool operator==(const FoldId &, const FoldId &) = default;
};

struct CapacityId {
  std::uint64_t value = 0u;
  [[nodiscard]] constexpr explicit operator bool() const { return value != 0u; }
  friend constexpr bool operator==(const CapacityId &,
                                   const CapacityId &) = default;
};

struct OutputId {
  std::uint64_t value = 0u;
  [[nodiscard]] constexpr explicit operator bool() const { return value != 0u; }
  friend constexpr bool operator==(const OutputId &,
                                   const OutputId &) = default;
};

struct RetryEpoch {
  std::uint64_t value = 0u;
  friend constexpr bool operator==(const RetryEpoch &,
                                   const RetryEpoch &) = default;
};

struct ShardRef {
  JobId job{};
  ShardId shard{};

  [[nodiscard]] constexpr explicit operator bool() const {
    return static_cast<bool>(job) && static_cast<bool>(shard);
  }

  friend constexpr bool operator==(const ShardRef &,
                                   const ShardRef &) = default;
};

struct RunKey {
  WorkId work{};
  bool sharded = false;
  ShardRef shard{};
  InputVersion input{};
  LogicalTime time{};
  CheckpointId checkpoint{};
  ProgramId program{};
  FoldId fold{};
  ::rund::evidence::Id numeric{};
  CapacityId capacity{};
  OutputId output{};

  [[nodiscard]] bool complete() const noexcept;

  friend bool operator==(const RunKey &lhs, const RunKey &rhs) noexcept;
};

struct RunAttempt {
  RunKey run{};
  RetryEpoch retry{};

  friend bool operator==(const RunAttempt &lhs, const RunAttempt &rhs) noexcept;
};

} // namespace rund::cluster
