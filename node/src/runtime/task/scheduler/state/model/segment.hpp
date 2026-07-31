#pragma once

#include <rund/reason.hpp>
#include <rund/task/operation/kind.hpp>

#include <cstddef>
#include <cstdint>

namespace rund::node {

struct TaskRecord;

struct LaneSegmentJob {
  std::uint64_t task_id = 0u;
  TaskRecord *record = nullptr;
  std::uint64_t logical_ticket = 0u;
  std::size_t canonical_index = 0u;
  bool split_primitive_packets = false;
};

struct LaneSegmentEffect {
  TaskRecord *record = nullptr;
  std::uint64_t task_id = 0u;
  std::uint64_t logical_ticket = 0u;
  std::size_t canonical_index = 0u;
  ::rund::detail::task::OperationKind terminal_kind =
      ::rund::detail::task::OperationKind::None;
  ReasonCode code = ReasonCode::Ok;
  bool terminal = false;
  bool trapped = false;
  ::rund::detail::task::OperationKind trap_kind =
      ::rund::detail::task::OperationKind::PrimitiveTrap;
  ReasonCode trap_code = ReasonCode::Ok;
};

struct LaneSegmentResultView {
  std::size_t completed = 0u;
  bool all_completed = false;
  bool has_trap_or_failure = false;
};

struct LaneCompletionGroup {
  std::uint64_t first_sequence = 0u;
  std::uint64_t expected_lanes = 0u;
  std::uint64_t observed_lanes = 0u;
  bool active = false;
};

} // namespace rund::node
