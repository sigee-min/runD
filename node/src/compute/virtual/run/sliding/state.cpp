#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

[[nodiscard]] bool control_range(const PipelineExecutionSnapshot &snapshot,
                                 const std::size_t roles,
                                 const std::uint64_t coordinates) noexcept {
  if ((roles != 2u && roles != 4u) || coordinates == 0u) {
    return false;
  }
  const std::uint64_t step = roles / residency::execution::BankCapacity;
  for (std::size_t slot = 0u; slot < roles; ++slot) {
    if (slot >= coordinates) {
      continue;
    }
    const std::size_t bank = slot % residency::execution::BankCapacity;
    const std::uint64_t parity_offset =
        slot / residency::execution::BankCapacity;
    const std::uint64_t turns = (coordinates - 1u - slot) / roles;
    const std::uint64_t first = snapshot.generation[bank] + parity_offset + 1u;
    if (first > std::numeric_limits<std::uint32_t>::max() ||
        turns > (std::numeric_limits<std::uint32_t>::max() - first) / step) {
      return false;
    }
  }
  return true;
}

[[maybe_unused, nodiscard]] Status
status_from(const rund::AccelCheck check) noexcept {
  return check.ok ? Status::success()
                  : Status::fail(
                        project_reason(check.reason, Reason::BackendFailed));
}

[[maybe_unused, nodiscard]] residency::execution::TerminalKind
terminal_from(const node::accel::detail::NativeTerminal terminal) noexcept {
  return terminal == node::accel::detail::NativeTerminal::Known
             ? residency::execution::TerminalKind::Known
             : residency::execution::TerminalKind::UnknownMayWrite;
}

void record_failure(SlidingProductRun &state, const Status failure,
                    const std::uint64_t coordinate) noexcept {
  if (failure) {
    return;
  }
  std::lock_guard lock{state.gate};
  if (state.failure || coordinate < state.failure_coordinate) {
    state.failure = failure;
    state.failure_coordinate = coordinate;
    state.result.failed_page = coordinate * state.run->frame_capacity;
  }
  state.ready.notify_all();
}

void record_service_fault(SlidingProductRun &state,
                          const ServiceFaultStage stage,
                          const std::uint32_t code,
                          const std::uint64_t coordinate, const Status failure,
                          const std::uint64_t key) noexcept {
  std::lock_guard lock{state.gate};
  if (state.service_fault.stage != ServiceFaultStage::None) {
    return;
  }
  state.service_fault = ServiceFault{
      .stage = stage,
      .code = code,
      .coordinate = coordinate,
      .reason = static_cast<std::uint32_t>(failure.reason()),
      .key = key,
  };
}

ServiceFault snapshot_service_fault(SlidingProductRun &state) noexcept {
  std::lock_guard lock{state.gate};
  return state.service_fault;
}

[[nodiscard]] bool reset_work(SlidingProductWork &work,
                              const std::uint64_t coordinate) noexcept {
  if (work.native || (work.projected && !work.native_released)) {
    return false;
  }
  work.projection = {};
  work.uses = {};
  work.output_phase.fill(OutputServicePhase::NeedDrain);
  work.native_status = Status::fail(Reason::PipelineInvalid);
  work.native_terminal = residency::execution::TerminalKind::Known;
  work.next_fetch = 0u;
  work.output_count = 0u;
  work.projected = false;
  work.promoted = false;
  work.terminaled = false;
  work.native_released = false;
  work.native_may_write = false;
  return coordinate < std::numeric_limits<std::uint64_t>::max();
}

[[nodiscard]] bool
discard_unissued_projection(SlidingProductWork &work) noexcept {
  if (!work.projected || work.native || work.promoted || work.terminaled ||
      work.native_released || work.native_may_write) {
    return false;
  }
  work.projection = {};
  work.uses = {};
  work.output_phase.fill(OutputServicePhase::NeedDrain);
  work.native_status = Status::fail(Reason::PipelineInvalid);
  work.native_terminal = residency::execution::TerminalKind::Known;
  work.next_fetch = 0u;
  work.output_count = 0u;
  work.projected = false;
  work.promoted = false;
  work.terminaled = false;
  work.native_released = false;
  work.native_may_write = false;
  return true;
}

} // namespace rund::compute::detail::sliding_product_detail
