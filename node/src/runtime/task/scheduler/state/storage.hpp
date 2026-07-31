#pragma once

#include <rund/host/io/fd.hpp>
#include <rund/host/random/seed.hpp>

#include "../state.hpp"
#include "../task/completion.hpp"

namespace rund::node {

#include "storage/batch.hpp"
#include "storage/evidence.hpp"
#include "storage/host/io.hpp"
#include "storage/identity.hpp"
#include "storage/lane.hpp"
#include "storage/plan.hpp"
#include "storage/reactor.hpp"
#include "storage/ready.hpp"
#include "storage/resources.hpp"

struct SchedulerState {
  SchedulerIdentityState identity{};
  SchedulerResourceState resources{};
  SchedulerHostIoState host_io{};
  SchedulerReadyState ready{};
  SchedulerReactorState reactor{};
  SchedulerLaneState lanes{};
  SchedulerEvidenceState evidence{};
  SchedulerPlanState plan{};
  SchedulerBatchState batches{};
  void *owner = nullptr;

  static constexpr std::size_t kInvalidTaskIndex =
      std::numeric_limits<std::size_t>::max();

  void RequireSequencer() const noexcept;

  [[nodiscard]] TaskRecord *Find(std::uint64_t id) noexcept;
  [[nodiscard]] const TaskRecord *Find(std::uint64_t id) const noexcept;
  [[nodiscard]] TaskRecord *FindAt(std::size_t index,
                                   std::uint64_t id) noexcept;
  [[nodiscard]] const TaskRecord *FindAt(std::size_t index,
                                         std::uint64_t id) const noexcept;
  [[nodiscard]] bool IndexTask(std::uint64_t id, std::size_t index) noexcept;
  void ForgetTask(std::uint64_t id) noexcept;
  [[nodiscard]] std::size_t IndexFor(std::uint64_t id) const noexcept;
  [[nodiscard]] std::size_t
  DefaultLaneIndexForTask(std::uint64_t id, std::size_t lane_count) noexcept;
  [[nodiscard]] std::size_t
  LaneIndexForTask(const TaskRecord *record,
                   std::size_t lane_count) const noexcept;
  [[nodiscard]] std::size_t
  LaneIndexForTask(std::uint64_t id, std::size_t lane_count) const noexcept;
  [[nodiscard]] bool EnqueueSpawn(const TaskRecord &record) noexcept;
  void EnqueueProgress(const TaskRecord &record) noexcept;

  [[nodiscard]] bool ScopeTerminal(std::uint64_t scope_id) const noexcept;
  [[nodiscard]] ReasonCode
  ScopeFailureCode(std::uint64_t scope_id) const noexcept;
  [[nodiscard]] ReasonCode FirstFailureCode() const noexcept;
};

} // namespace rund::node
