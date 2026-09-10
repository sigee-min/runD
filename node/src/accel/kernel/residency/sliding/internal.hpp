#pragma once

#include "../../prepared/interface/api.hpp"

#include "../../../backend/ops/table.hpp"
#include "../../../clock.hpp"
#include "../../prepared/evidence.hpp"
#include "../../prepared/model.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <utility>

namespace rund::node::accel::detail::prepared::sliding {

enum class SlotPhase : std::uint8_t {
  Done,
  Idle,
  Projecting,
  Submitting,
  Submitted,
  Returning,
};

struct State;

struct Service final : std::enable_shared_from_this<Service> {
  using Work = void (*)(const std::shared_ptr<State> &) noexcept;

  std::mutex gate{};
  std::condition_variable ready{};
  std::shared_ptr<State> pending{};
  std::thread worker{};
  Work work{};
  Work finish{};
  bool working{};
  bool again{};
  bool stop{};

  ~Service();
  [[nodiscard]] bool Start(Work, Work) noexcept;
  void Schedule(const std::shared_ptr<State> &) noexcept;
  void Shutdown() noexcept;
};

struct Slot final {
  State *state{};
  PreparedKernelPipeline pipeline{};
  PreparedResidencySlidingSelection selection{};
  KernelResult inline_result{};
  std::atomic_bool submit_returned{true};
  std::uint64_t coordinate{};
  std::uint64_t turn{};
  std::uint8_t slot{};
  SlotPhase phase{SlotPhase::Done};
  bool inline_terminal{};
  bool same_thread_inline{};
};

struct ProjectHandoff final {
  PreparedResidencySlidingProject callback{};
  void *user{};
  std::uint64_t coordinate{};
  std::uint64_t turn{};
  std::uint8_t slot{};
};

struct Admission final {
  std::array<prepared::PipelineState *, ResidencySlidingCapacity> pipelines{};
  BackendResidencySlidingCapability capability{};
  std::size_t role_count{};
};

struct State final : std::enable_shared_from_this<State> {
  std::mutex gate{};
  std::condition_variable pump_ready{};
  rund::AccelContext context{};
  std::array<PreparedResidencySlidingRole, ResidencySlidingCapacity> roles{};
  std::array<prepared::PipelineState *, ResidencySlidingCapacity> pipelines{};
  std::array<Slot, ResidencySlidingCapacity> slots{};
  PreparedResidencySlidingRequest request{};
  BackendResidencySlidingCapability capability{};
  std::shared_ptr<Service> service{};
  std::shared_ptr<State> active_owner{};
  std::shared_ptr<State> quarantine{};
  std::size_t role_count{};
  std::uint64_t accepted{};
  std::uint64_t released{};
  std::uint64_t queue_calls{};
  std::uint64_t inflight{};
  std::uint64_t inflight_peak{};
  std::uint64_t external_calls{};
  std::uint64_t first_failure{std::numeric_limits<std::uint64_t>::max()};
  rund::AccelCheck failure{true, "ok"};
  bool active{};
  bool pumping{};
  bool pump_pending{};
  bool failed{};
  bool unknown{};
  bool final_sent{};
  std::atomic_bool service_queued{};

  ~State();
};

extern thread_local Slot *SubmittingSlot;
extern thread_local State *PumpingState;

void service_pump(const std::shared_ptr<State> &) noexcept;
void service_finish(const std::shared_ptr<State> &) noexcept;
void schedule_service(State &) noexcept;

[[nodiscard]] rund::AccelCheck
admit(const rund::AccelContext &, std::span<const PreparedResidencySlidingRole>,
      ResidencySlidingMemory, Admission &) noexcept;

[[nodiscard]] bool valid_memory(const BackendResidencySlidingCapability &,
                                ResidencySlidingMemory) noexcept;
[[nodiscard]] bool
same_capability(const BackendResidencySlidingCapability &,
                const BackendResidencySlidingCapability &) noexcept;
[[nodiscard]] bool
valid_selection(const PreparedResidencySlidingSelection &) noexcept;
[[nodiscard]] bool same_roles(const State &,
                              const PreparedResidencySlidingRequest &) noexcept;
void release_claims_known(State &) noexcept;
void record_failure(State &, std::uint64_t, rund::AccelCheck) noexcept;

void emit_final(const std::shared_ptr<State> &) noexcept;
void complete_native(void *, KernelResult) noexcept;
void fail_projection(State &, Slot &, rund::AccelCheck) noexcept;

[[nodiscard]] bool service_returned(const std::shared_ptr<State> &,
                                    std::size_t) noexcept;
[[nodiscard]] bool begin_projection(State &, std::size_t,
                                    ProjectHandoff &) noexcept;
[[nodiscard]] bool
complete_projection(State &, std::size_t, const ProjectHandoff &,
                    PreparedResidencySlidingProjection,
                    const PreparedResidencySlidingSelection &) noexcept;
void submit_slot(const std::shared_ptr<State> &, std::size_t) noexcept;
void pump(const std::shared_ptr<State> &) noexcept;

} // namespace rund::node::accel::detail::prepared::sliding
