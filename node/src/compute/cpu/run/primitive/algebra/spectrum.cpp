#include "local.hpp"

#include "../../../graph.hpp"
#include "../../../scratch.hpp"

#include "../../../../../accel/spectrum.hpp"

#include <kernel/program/compute/spectrum/reference.hpp>

namespace rund::compute::detail {

Status run_spectrum(PrimitiveContext &context) {
  const auto &plan = std::get<kernel::SpectrumPlan>(context.primitive.plan);
  const bool has_vectors = plan.vector_count != 0u;
  const RawCpuBuffer &input = context.port(0u);
  const RawCpuBuffer &values = context.port(1u);
  const RawCpuBuffer *const vectors = has_vectors ? &context.port(2u) : nullptr;
  const RawCpuBuffer &status = context.ports.back();
  if (!input || !values || !status || (vectors != nullptr && !*vectors)) {
    return finish_cpu_primitive(false, "compute_spectrum_invalid");
  }
  kernel::SpectrumResult result{};
  if (plan.element_bytes == sizeof(kernel::i64)) {
    CpuSpectrumScratchView<kernel::i64> scratch{};
    if (!cpu_spectrum_scratch(cpu_step_scratch(context.run, context.step), plan,
                              scratch)) {
      return finish_cpu_primitive(false, "compute_spectrum_invalid");
    }
    result = kernel::ReferenceSpectrumScratchI64(
        reinterpret_cast<const kernel::i64 *>(input.data),
        reinterpret_cast<kernel::i64 *>(values.data),
        vectors == nullptr ? nullptr
                           : reinterpret_cast<kernel::i64 *>(vectors->data),
        reinterpret_cast<kernel::u32 *>(status.data), plan, scratch.matrix,
        scratch.vectors, scratch.values, scratch.order, scratch.u);
  } else {
    CpuSpectrumScratchView<kernel::i32> scratch{};
    if (!cpu_spectrum_scratch(cpu_step_scratch(context.run, context.step), plan,
                              scratch)) {
      return finish_cpu_primitive(false, "compute_spectrum_invalid");
    }
    result = kernel::ReferenceSpectrumScratchI32(
        reinterpret_cast<const kernel::i32 *>(input.data),
        reinterpret_cast<kernel::i32 *>(values.data),
        vectors == nullptr ? nullptr
                           : reinterpret_cast<kernel::i32 *>(vectors->data),
        reinterpret_cast<kernel::u32 *>(status.data), plan, scratch.matrix,
        scratch.vectors, scratch.values, scratch.order, scratch.u);
  }
  record_cpu_algebra_status(context.run, Primitive::Spectrum,
                            result.failed_batches,
                            static_cast<std::uint32_t>(result.first_status));
  return finish_cpu_primitive(result.ok, result.reason);
}

} // namespace rund::compute::detail
