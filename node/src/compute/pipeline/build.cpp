#include <rund/compute/pipeline.hpp>
#include <rund/compute/resource/plan.hpp>

#include "../size.hpp"
#include "../type.hpp"
#include "assembly/internal.hpp"

#include <limits>
#include <memory>
#include <new>

namespace rund::compute::detail {

std::shared_ptr<PipelineBuildState>
make_pipeline(const std::shared_ptr<DeviceState> &device) noexcept {
  try {
    auto state = std::make_shared<PipelineBuildState>();
    state->device = device;
    state->steps.reserve(PipelineStepCapacity);
    state->state_pairs.reserve(PipelineLeafCapacity);
    state->publications.reserve(PipelineLeafCapacity);
    state->internals.reserve(PipelineLeafCapacity);
    state->window_controls.reserve(PipelineStepCapacity);
    state->nested_windows.reserve(PipelineStepCapacity);
    if (device == nullptr) {
      state->failure = Reason::DeviceInvalid;
    }
    return state;
  } catch (const std::bad_alloc &) {
    return {};
  }
}

void append_pipeline(const std::shared_ptr<PipelineBuildState> &build,
                     const std::shared_ptr<ProgramState> &program,
                     const std::span<const ResourceView> inputs,
                     const std::span<const ResourceView> outputs) noexcept {
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
  const std::size_t binding_capacity = build->nested_windows.empty()
                                           ? PipelineBindingCapacity
                                           : PipelineRouteBindingCapacity;
  const std::size_t route_capacity = build->nested_windows.empty()
                                         ? PipelineIterationCapacity
                                         : PipelineRouteCapacity;
  if (build->logical_step_count >= PipelineStepCapacity ||
      build->steps.size() >= route_capacity ||
      inputs.size() > PipelineLeafCapacity ||
      outputs.size() > PipelineLeafCapacity - inputs.size() ||
      build->binding_count > binding_capacity ||
      inputs.size() + outputs.size() >
          binding_capacity - build->binding_count) {
    build->failure = Reason::PipelineCapacity;
    return;
  }
  try {
    PipelineBuildStep step{};
    step.logical_step = static_cast<std::uint32_t>(build->logical_step_count);
    step.program = program;
    step.inputs.reserve(inputs.size());
    step.outputs.reserve(outputs.size());
    for (const ResourceView &view : inputs) {
      if (view.access != ResourceAccess::Read) {
        build->failure = Reason::BindingInvalid;
        return;
      }
      PipelineBinding binding{};
      const Status routed = route(*build, view, binding);
      if (!routed) {
        build->failure = routed.reason();
        return;
      }
      step.inputs.push_back(std::move(binding));
    }
    for (const ResourceView &view : outputs) {
      if (view.access != ResourceAccess::Write) {
        build->failure = Reason::BindingInvalid;
        return;
      }
      PipelineBinding binding{};
      const Status routed = route(*build, view, binding);
      if (!routed) {
        build->failure = routed.reason();
        return;
      }
      step.outputs.push_back(std::move(binding));
    }
    build->binding_count += inputs.size() + outputs.size();
    build->steps.push_back(std::move(step));
    ++build->logical_step_count;
    changed(*build);
  } catch (const std::bad_alloc &) {
    build->failure = Reason::PipelineCapacity;
  }
}
void append_pipeline_state(const std::shared_ptr<PipelineBuildState> &build,
                           const std::shared_ptr<BufferState> &published,
                           const std::shared_ptr<BufferState> &pending,
                           const Type type, const FixedFormat format) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || has_seed(*build)) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  if (published == nullptr || pending == nullptr || published == pending) {
    build->failure = published == pending ? Reason::BindingDuplicate
                                          : Reason::BindingInvalid;
    return;
  }
  if (published->device != build->device || pending->device != build->device) {
    build->failure = Reason::BindingDeviceMismatch;
    return;
  }
  if (!valid_type(type) || published->type != type || pending->type != type) {
    build->failure = Reason::BindingTypeMismatch;
    return;
  }
  const std::size_t element_bytes = type_bytes(type);
  std::size_t expected_bytes = 0u;
  if (element_bytes == 0u || published->count != pending->count ||
      published->bytes != pending->bytes ||
      !size::multiply(published->count, element_bytes, expected_bytes) ||
      published->bytes != expected_bytes ||
      published->physical_bytes < published->bytes ||
      pending->physical_bytes < pending->bytes) {
    build->failure = Reason::ShapeMismatch;
    return;
  }
  for (const PipelineBuildStatePair &pair : build->state_pairs) {
    if (pair.published.buffer == published ||
        pair.published.buffer == pending || pair.pending.buffer == published ||
        pair.pending.buffer == pending) {
      build->failure = Reason::BindingDuplicate;
      return;
    }
  }
  if (build->state_pairs.size() >= PipelineLeafCapacity) {
    build->failure = Reason::PipelineCapacity;
    return;
  }
  try {
    const auto binding = [&](const std::shared_ptr<BufferState> &buffer,
                             const ResourceAccess access) {
      return PipelineBinding{.buffer = buffer,
                             .type = type,
                             .format = format,
                             .count = buffer->count,
                             .stride = 1u,
                             .element_bytes = element_bytes,
                             .alignment = element_bytes,
                             .backing_bytes = buffer->bytes,
                             .access = access};
    };
    build->state_pairs.push_back(PipelineBuildStatePair{
        .published = binding(published, ResourceAccess::Read),
        .pending = binding(pending, ResourceAccess::Write),
    });
    changed(*build);
  } catch (const std::bad_alloc &) {
    build->failure = Reason::PipelineCapacity;
  }
}

void configure_pipeline_profile(
    const std::shared_ptr<PipelineBuildState> &build,
    const PipelineProfile profile) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  if (profile != PipelineProfile::None && profile != PipelineProfile::Steps) {
    build->failure = Reason::ProfileInvalid;
    return;
  }
  build->profile = profile;
  changed(*build);
}

void configure_pipeline_sealed_repetitions(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::size_t repetitions) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || has_seed(*build) ||
      build->sealed_repetitions_configured || repetitions == 0u ||
      repetitions > PipelineSealedRepetitionCapacity ||
      repetitions > std::numeric_limits<std::uint32_t>::max()) {
    build->failure =
        repetitions == 0u || repetitions > PipelineSealedRepetitionCapacity
            ? Reason::PipelineCapacity
            : Reason::PipelineInvalid;
    return;
  }
  build->sealed_repetitions = static_cast<std::uint32_t>(repetitions);
  build->sealed_repetitions_configured = true;
  changed(*build);
}

void commit_pipeline(
    const std::shared_ptr<PipelineBuildState> &build) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || build->state_pairs.empty()) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  build->commit = true;
  build->sealed = true;
}

void seed_pipeline(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<StateSnapshotState> &snapshot) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || build->commit || build->steps.empty() ||
      build->state_pairs.empty() || snapshot == nullptr || has_seed(*build)) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  build->seed = snapshot;
}

void seed_pipeline(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<PipelinePublicationState> &publication) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || build->commit || build->steps.empty() ||
      build->state_pairs.empty() || publication == nullptr ||
      has_seed(*build)) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  build->device_seed = publication;
}

void seed_pipeline(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || build->commit || build->steps.empty() ||
      build->state_pairs.empty() || storage == nullptr || has_seed(*build)) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  build->storage_seed = storage;
}

} // namespace rund::compute::detail
