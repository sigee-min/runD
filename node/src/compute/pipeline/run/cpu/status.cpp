#include "../../../cpu/graph.hpp"
#include "../../../cpu/state/program.hpp"
#include "internal.hpp"

#include "../../state.hpp"

#include "../../../backend.hpp"
#include "../../../status.hpp"

#include <rund/counter.hpp>

#include <cstdint>

namespace rund::compute::detail {

std::uint64_t cpu_program_status_entries(const ProgramState &program) noexcept {
  if (program.device == nullptr || program.device->backend != Backend::Cpu ||
      program.cpu_graph == nullptr || program.cpu_graph->runtime == nullptr) {
    return 0u;
  }
  std::uint64_t total = 0u;
  for (const CpuRuntimeStep &runtime_step : program.cpu_graph->runtime->steps) {
    const auto *const primitive =
        std::get_if<CpuRuntimePrimitive>(&runtime_step);
    if (primitive == nullptr) {
      continue;
    }
    std::uint64_t count = 0u;
    switch (primitive->kind) {
    case Primitive::Factor:
      if (const auto *const plan =
              std::get_if<kernel::FactorPlan>(&primitive->plan)) {
        count = plan->status_count;
      }
      break;
    case Primitive::Solve:
      if (const auto *const plan =
              std::get_if<kernel::SolvePlan>(&primitive->plan)) {
        count = plan->status_count;
      }
      break;
    case Primitive::Spectrum:
      if (const auto *const plan =
              std::get_if<kernel::SpectrumPlan>(&primitive->plan)) {
        count = plan->status_count;
      }
      break;
    default:
      break;
    }
    total = ::rund::detail::counter::SaturatingAdd(total, count);
  }
  return total;
}

} // namespace rund::compute::detail
