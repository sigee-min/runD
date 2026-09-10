#include <rund/compute/pipeline.hpp>
#include <rund/compute/resource/plan.hpp>

#include "../../program/output.hpp"
#include "../../size.hpp"
#include "../../type.hpp"
#include "../output.hpp"
#include "../state.hpp"
#include "internal.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <new>

namespace rund::compute::detail {

namespace {

void append_recurrence(const std::shared_ptr<PipelineBuildState> &build,
                       const std::shared_ptr<ProgramState> &program,
                       const std::span<const ResourceView> inputs,
                       const std::span<const ResourceView> outputs,
                       const std::size_t iterations,
                       const bool write_each_iteration,
                       const ResourceView *const resident,
                       const std::size_t maximum, const std::size_t tile,
                       const std::size_t terminal,
                       const std::uint32_t expected) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || has_seed(*build)) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  if (program == nullptr) {
    build->failure = Reason::ProgramInvalid;
    return;
  }
  const std::size_t controls = resident == nullptr ? 0u : 2u;
  const std::size_t step_bindings = inputs.size() + controls + outputs.size();
  const std::size_t route_capacity = build->nested_windows.empty()
                                         ? PipelineIterationCapacity
                                         : PipelineRouteCapacity;
  const std::size_t binding_capacity = build->nested_windows.empty()
                                           ? PipelineBindingCapacity
                                           : PipelineRouteBindingCapacity;
  std::size_t expanded_bindings = 0u;
  if (iterations == 0u || iterations > PipelineIterationCapacity ||
      outputs.empty() || outputs.size() > inputs.size() ||
      (terminal != NoWindowTerminal &&
       (terminal >= outputs.size() || outputs[terminal].type != Type::U32 ||
        inputs[terminal].type != Type::U32 || outputs[terminal].count != 1u ||
        inputs[terminal].count != 1u)) ||
      inputs.size() + controls > PipelineLeafCapacity ||
      outputs.size() > PipelineLeafCapacity - inputs.size() - controls ||
      build->logical_step_count >= PipelineStepCapacity ||
      build->steps.size() > route_capacity - iterations ||
      !size::multiply(step_bindings, iterations, expanded_bindings) ||
      build->binding_count > binding_capacity ||
      expanded_bindings > binding_capacity - build->binding_count) {
    build->failure = Reason::PipelineCapacity;
    return;
  }
  const bool write_each = write_each_iteration && iterations > 1u;
  const std::size_t first = build->steps.size();
  const std::size_t old_bindings = build->binding_count;
  const std::size_t old_internals = build->internals.size();
  const std::size_t old_publications = build->publications.size();
  const std::size_t old_window_controls = build->window_controls.size();
  try {
    std::vector<PipelineBinding> scratch;
    std::vector<PipelineBinding> alternate;
    std::vector<PipelineBinding> history;
    std::vector<std::size_t> history_counts;
    scratch.reserve(write_each ? 0u : outputs.size());
    alternate.reserve(resident == nullptr ? 0u : outputs.size());
    history.reserve(write_each ? outputs.size() : 0u);
    history_counts.reserve(write_each ? outputs.size() : 0u);
    for (const ResourceView &output : outputs) {
      if (output.access != ResourceAccess::Write) {
        build->failure = Reason::BindingInvalid;
        return;
      }
      const std::size_t bytes = type_bytes(output.type);
      if (bytes == 0u || !size::multiply(output.count, bytes) ||
          output.element_bytes != bytes) {
        build->failure = Reason::ShapeMismatch;
        return;
      }
      if (write_each) {
        if (output.count % iterations != 0u) {
          build->failure = Reason::ShapeMismatch;
          return;
        }
        history.push_back(bind(output));
        history_counts.push_back(output.count / iterations);
        continue;
      }
    }
    if (std::any_of(inputs.begin(), inputs.end(), [](const ResourceView &view) {
          return view.access != ResourceAccess::Read;
        })) {
      build->failure = Reason::BindingInvalid;
      return;
    }
    if (resident != nullptr &&
        (resident->access != ResourceAccess::Read ||
         resident->type != Type::U32 || resident->count != 1u ||
         maximum == 0u || tile == 0u || tile > maximum ||
         build->window_controls.size() >=
             PipelineBuildWindowControlOrdinal::unassigned)) {
      build->failure = Reason::BindingInvalid;
      return;
    }
    auto projection = project_outputs(*program, outputs.size());
    if (!projection) {
      build->failure = projection.reason();
      return;
    }
    for (std::size_t logical = 0u; logical < outputs.size(); ++logical) {
      const std::uint32_t physical = projection->logical_to_physical[logical];
      if (physical >= projection->physical_count) {
        build->failure = Reason::PipelineInvalid;
        return;
      }
      const std::uint32_t canonical = projection->physical_sources[physical];
      if (canonical >= outputs.size() ||
          !same_view(outputs[logical], outputs[canonical])) {
        build->failure = Reason::BindingAliasUnsupported;
        return;
      }
    }
    if (!write_each) {
      for (std::size_t physical = 0u; physical < projection->physical_count;
           ++physical) {
        const std::uint32_t logical = projection->physical_sources[physical];
        if (logical >= outputs.size()) {
          build->failure = Reason::PipelineInvalid;
          return;
        }
        const ResourceView &output = outputs[logical];
        const auto owner = static_cast<std::uint32_t>(build->internals.size());
        build->internals.push_back(PipelineInternal{.type = output.type,
                                                    .format = output.format,
                                                    .count = output.count});
        scratch.push_back(bind(owner, build->internals.back(),
                               ResourceAccess::Write, 0u, output.count, true));
        if (resident != nullptr) {
          const auto second =
              static_cast<std::uint32_t>(build->internals.size());
          build->internals.push_back(PipelineInternal{
              .type = output.type,
              .format = output.format,
              .count = output.count,
          });
          alternate.push_back(bind(second, build->internals.back(),
                                   ResourceAccess::Write, 0u, output.count,
                                   true));
        }
      }
    }
    std::vector<PipelineBinding> current;
    current.reserve(inputs.size());
    for (const ResourceView &input : inputs) {
      PipelineBinding binding{};
      const Status routed = route(*build, input, binding);
      if (!routed) {
        build->failure = routed.reason();
        return;
      }
      current.push_back(std::move(binding));
    }

    std::uint32_t ordinal_owner = PipelineBinding::external;
    PipelineBuildWindowControlOrdinal window_control{};
    if (resident != nullptr) {
      window_control.value =
          static_cast<std::uint32_t>(build->window_controls.size());
      ordinal_owner = static_cast<std::uint32_t>(build->internals.size());
      build->internals.push_back(PipelineInternal{
          .type = Type::U32,
          .count = iterations,
          .fill = PipelineFill::Ordinal,
      });
    }

    std::vector<PipelineBinding> final;
    final.reserve(write_each ? 0u : outputs.size());
    if (!write_each) {
      for (const ResourceView &output : outputs) {
        final.push_back(bind(output));
      }
    }
    std::vector<PipelineBinding> each;
    each.reserve(write_each ? outputs.size() : 0u);
    std::vector<PipelineBinding> projected_destination;
    projected_destination.reserve(write_each ? 0u : outputs.size());
    for (std::size_t iteration = 0u; iteration < iterations; ++iteration) {
      each.clear();
      projected_destination.clear();
      if (write_each) {
        for (std::size_t index = 0u; index < history.size(); ++index) {
          PipelineBinding binding = history[index];
          std::size_t first_element = 0u;
          std::size_t offset = 0u;
          if (!size::multiply(iteration, history_counts[index],
                              first_element) ||
              !size::multiply(first_element, binding.stride, first_element) ||
              !size::add(binding.offset, first_element, offset)) {
            build->failure = Reason::ShapeMismatch;
            return;
          }
          binding.offset = offset;
          binding.count = history_counts[index];
          each.push_back(std::move(binding));
        }
      }
      const bool final_bank =
          resident == nullptr && ((iterations - iteration) & 1u) != 0u;
      const std::span<const PipelineBinding> physical_destination = [&] {
        if (write_each) {
          return std::span<const PipelineBinding>{each};
        }
        if (final_bank) {
          return std::span<const PipelineBinding>{final};
        }
        return resident != nullptr && (iteration & 1u) != 0u
                   ? std::span<const PipelineBinding>{alternate}
                   : std::span<const PipelineBinding>{scratch};
      }();
      const std::span<const PipelineBinding> destination = [&] {
        if (write_each || final_bank) {
          return physical_destination;
        }
        for (std::size_t logical = 0u; logical < outputs.size(); ++logical) {
          const std::uint32_t physical =
              projection->logical_to_physical[logical];
          if (physical >= physical_destination.size()) {
            return std::span<const PipelineBinding>{};
          }
          projected_destination.push_back(physical_destination[physical]);
        }
        return std::span<const PipelineBinding>{projected_destination};
      }();
      if (destination.size() != outputs.size()) {
        build->failure = Reason::PipelineInvalid;
        return;
      }
      PipelineBuildStep step{};
      step.program = program;
      step.logical_step = static_cast<std::uint32_t>(build->logical_step_count);
      step.iteration = static_cast<std::uint32_t>(iteration);
      step.iteration_bound = static_cast<std::uint32_t>(iterations);
      step.window_control = window_control;
      step.writes_each_iteration = write_each;
      step.inputs.reserve(inputs.size() + controls);
      step.outputs.reserve(outputs.size());
      for (PipelineBinding binding : current) {
        binding.access = ResourceAccess::Read;
        step.inputs.push_back(std::move(binding));
      }
      if (resident != nullptr) {
        step.inputs.push_back(bind(*resident));
        step.inputs.push_back(bind(ordinal_owner,
                                   build->internals[ordinal_owner],
                                   ResourceAccess::Read, iteration, 1u));
      }
      for (PipelineBinding binding : destination) {
        binding.access = ResourceAccess::Write;
        binding.hidden = !write_each && (resident != nullptr || !final_bank);
        step.outputs.push_back(std::move(binding));
      }
      build->steps.push_back(std::move(step));
      for (std::size_t index = 0u; index < outputs.size(); ++index) {
        current[index] = destination[index];
        current[index].access = ResourceAccess::Read;
      }
    }
    if (resident != nullptr) {
      build->window_controls.push_back(PipelineBuildWindowControl{
          .ordinary_step = {.value = first},
          .count_input = inputs.size(),
          .maximum = maximum,
          .tile = tile,
          .terminal = terminal,
          .expected = expected,
      });
      for (std::size_t physical = 0u; physical < projection->physical_count;
           ++physical) {
        const std::uint32_t logical = projection->physical_sources[physical];
        build->publications.push_back(PipelineBuildTerminalPublication{
            .edge =
                {
                    .target = final[logical],
                    .control = window_control,
                    .output = {.value = logical},
                },
        });
      }
    }
    build->binding_count += expanded_bindings;
    ++build->logical_step_count;
    changed(*build);
  } catch (const std::bad_alloc &) {
    build->steps.resize(first);
    build->internals.resize(old_internals);
    build->publications.resize(old_publications);
    build->window_controls.resize(old_window_controls);
    build->binding_count = old_bindings;
    build->failure = Reason::PipelineCapacity;
  }
}

} // namespace

void append_pipeline_repeat(const std::shared_ptr<PipelineBuildState> &build,
                            const std::shared_ptr<ProgramState> &program,
                            const std::span<const ResourceView> inputs,
                            const std::span<const ResourceView> outputs,
                            const std::size_t iterations) noexcept {
  append_recurrence(build, program, inputs, outputs, iterations, false, nullptr,
                    0u, 0u, NoWindowTerminal, 1u);
}

void append_pipeline_repeat_each(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<ProgramState> &program,
    const std::span<const ResourceView> inputs,
    const std::span<const ResourceView> outputs,
    const std::size_t iterations) noexcept {
  append_recurrence(build, program, inputs, outputs, iterations, true, nullptr,
                    0u, 0u, NoWindowTerminal, 1u);
}

void append_pipeline_windows(const std::shared_ptr<PipelineBuildState> &build,
                             const std::shared_ptr<ProgramState> &program,
                             const ResourceView &resident,
                             const std::span<const ResourceView> inputs,
                             const std::span<const ResourceView> outputs,
                             const std::size_t maximum, const std::size_t tile,
                             const std::size_t terminal,
                             const std::uint32_t expected) noexcept {
  std::size_t rounded = 0u;
  if (tile == 0u || maximum == 0u || tile > maximum ||
      maximum > std::numeric_limits<std::uint32_t>::max() ||
      !size::add(maximum, tile - 1u, rounded)) {
    if (build != nullptr && build->failure == Reason::Ok) {
      build->failure = Reason::PipelineCapacity;
    }
    return;
  }
  const std::size_t iterations = rounded / tile;
  append_recurrence(build, program, inputs, outputs, iterations, false,
                    &resident, maximum, tile, terminal, expected);
}
} // namespace rund::compute::detail
