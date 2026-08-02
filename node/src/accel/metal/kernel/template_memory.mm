#include "template_memory.hpp"

#include "local.hpp"
#include "pipeline/build.hpp"

#include <rund/counter.hpp>

#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] std::uint64_t
AddTemplateBytes(const std::uint64_t left, const std::uint64_t right) noexcept {
  return ::rund::detail::counter::SaturatingAdd(left, right);
}

template <typename T>
[[nodiscard]] std::uint64_t
VectorTemplateBytes(const std::vector<T> &values) noexcept {
  return ::rund::detail::counter::SaturatingMultiply(
      static_cast<std::uint64_t>(values.capacity()), sizeof(T));
}

[[nodiscard]] std::uint64_t
MetalMapTemplateBytes(const MetalMapTemplateResources &prepared) noexcept {
  std::uint64_t bytes = sizeof(MetalMapTemplateResources);
  bytes = AddTemplateBytes(bytes, VectorTemplateBytes(prepared.input_plans));
  bytes = AddTemplateBytes(bytes, VectorTemplateBytes(prepared.input_strides));
  bytes = AddTemplateBytes(bytes, VectorTemplateBytes(prepared.output_strides));
  return AddTemplateBytes(bytes, VectorTemplateBytes(prepared.checks));
}

[[nodiscard]] PreparedMemory
TemplateMemory(const std::uint64_t bytes) noexcept {
  return PreparedMemory{
      .current = bytes, .peak = bytes, .cumulative = bytes, .budget = bytes};
}

[[nodiscard]] PreparedMemory InvalidTemplateMemory() noexcept {
  return TemplateMemory(std::numeric_limits<std::uint64_t>::max());
}

[[nodiscard]] PreparedMemory ObserveMetalProgramTemplate(
    const MetalKernelProgramTemplate &program) noexcept {
  if (program.signature == nullptr || program.signature->steps == nullptr ||
      program.signature->step_count != program.steps.size()) {
    return InvalidTemplateMemory();
  }
  std::uint64_t bytes = AddTemplateBytes(sizeof(MetalKernelProgramTemplate),
                                         VectorTemplateBytes(program.steps));
  for (std::size_t index = 0u; index < program.steps.size(); ++index) {
    const BoundStep &bound = program.signature->steps[index];
    const std::shared_ptr<const void> &immutable =
        program.steps[index].immutable;
    if (bound.step == nullptr || immutable == nullptr) {
      return InvalidTemplateMemory();
    }
    if (bound.step->kind() == rund::kernel::NodeKind::Map) {
      bytes = AddTemplateBytes(
          bytes,
          MetalMapTemplateBytes(*static_cast<const MetalMapTemplateResources *>(
              immutable.get())));
    } else {
      // Primitive pipeline handles are inline in this fixed wrapper. Their
      // driver-private allocations are intentionally outside host telemetry.
      bytes = AddTemplateBytes(bytes, sizeof(MetalKernelImmutablePipelines));
    }
  }
  return TemplateMemory(bytes);
}

[[nodiscard]] PreparedMemory ObserveMetalRecurrenceTemplate(
    const MetalMapRecurrenceTemplate &recurrence) noexcept {
  if (recurrence.signature == nullptr ||
      recurrence.signature->steps == nullptr ||
      recurrence.signature->step_count != 1u ||
      recurrence.prepared == nullptr) {
    return InvalidTemplateMemory();
  }
  std::uint64_t bytes = sizeof(MetalMapRecurrenceTemplate);
  bytes = AddTemplateBytes(bytes, MetalMapTemplateBytes(*recurrence.prepared));
  return TemplateMemory(bytes);
}

} // namespace
#endif

PreparedMemory
ObserveMetalPipelineTemplate(const void *const prepared) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  if (prepared == nullptr) {
    return InvalidTemplateMemory();
  }
  switch (MetalKernelTemplateKindOf(prepared)) {
  case MetalKernelTemplateKind::Program:
    return ObserveMetalProgramTemplate(
        *static_cast<const MetalKernelProgramTemplate *>(prepared));
  case MetalKernelTemplateKind::MapRecurrence:
    return ObserveMetalRecurrenceTemplate(
        *static_cast<const MetalMapRecurrenceTemplate *>(prepared));
  }
  return InvalidTemplateMemory();
#else
  (void)prepared;
  return {};
#endif
}

} // namespace rund::node::accel::detail
