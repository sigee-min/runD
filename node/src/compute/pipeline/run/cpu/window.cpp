#include "internal.hpp"

#include "../../state.hpp"

#include "../../../cpu/view.hpp"

#include <rund/counter.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace rund::compute::detail {

Status prepare_cpu_pipeline_window(const CpuPipelinePublicationContext &context,
                                   PipelineWindows &windows,
                                   const std::uint16_t window,
                                   const std::uint32_t iteration,
                                   ControlStats &stats, bool &active) noexcept {
  active = true;
  if (window == 0u) {
    return Status::success();
  }
  const PipelineWindow *const descriptor = windows.find(window);
  if (descriptor == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  PipelineWindowProgress &progress = windows.progress(window - 1u);
  std::uint32_t item_count{};
  CpuView count{};
  const Status count_ready = resolve_cpu_pipeline_publication_view(
      context, descriptor->control.count, count);
  if (!count_ready || count.data == nullptr || count.footprint.count != 1u ||
      count.footprint.width != sizeof(item_count)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::memcpy(&item_count, count.data, sizeof(item_count));
  if (item_count > descriptor->control.maximum) {
    stats.overflow_ordinal = descriptor->control.maximum;
    return Status::fail(Reason::BoundedCountInvalid);
  }
  const std::uint64_t base =
      static_cast<std::uint64_t>(iteration) * descriptor->control.tile;
  bool terminal = false;
  if (descriptor->control.terminal_publication !=
      std::numeric_limits<std::uint32_t>::max()) {
    if (descriptor->control.terminal_publication >=
        context.publications.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const auto *terminal_plan = std::get_if<PipelineTerminalPublicationPlan>(
        &context.publications[descriptor->control.terminal_publication]);
    const std::uint32_t state_index = static_cast<std::uint32_t>(window - 1u);
    if (terminal_plan == nullptr || terminal_plan->state != state_index ||
        progress.current > PipelineWindow::second) {
      return Status::fail(Reason::PipelineInvalid);
    }
    CpuView view{};
    const Status terminal_ready = resolve_cpu_pipeline_publication_view(
        context, terminal_plan->sources[progress.current], view);
    if (!terminal_ready || view.footprint.count != 1u ||
        view.footprint.width != sizeof(std::uint32_t)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    std::uint32_t observed{};
    std::memcpy(&observed, view.data, sizeof(observed));
    terminal = observed == descriptor->control.expected;
  }
  if (!progress.stopped && base < item_count && !terminal) {
    return Status::success();
  }

  active = false;
  if (!progress.stopped) {
    const Status sealed =
        seal_cpu_resident(context, window - 1u, *descriptor, progress);
    if (!sealed) {
      return sealed;
    }
    progress.stopped = true;
  }
  ::rund::detail::counter::Accumulate(stats.skipped_iteration_count, 1u);
  return Status::success();
}

Status prepare_cpu_pipeline_window(PipelineState &state,
                                   const std::size_t index,
                                   bool &active) noexcept {
  if (index >= state.steps.size()) {
    active = false;
    return Status::fail(Reason::PipelineInvalid);
  }
  const PipelineStep &step = state.steps[index];
  return prepare_cpu_pipeline_window(cpu_pipeline_publication_context(state),
                                     state.windows, step.window, step.iteration,
                                     state.stats.control, active);
}

} // namespace rund::compute::detail
