#include "local.hpp"

#include "../../context/internal.hpp"
namespace rund::node::accel::detail {

bool BuildStepBinds(const KernelExecutionStep& step,
                    const RunBinds& run_binds,
                    StepBinds& out) {
  const rund::kernel::ExecutionMetadata& metadata = step.artifact.metadata;
  if (!step.graph_binding_indices_ok) {
    return false;
  }
  const BindingSource source = BindingSourceFor(run_binds);
  if (!BindingSourceValid(source)) {
    return false;
  }

  out = {};
  out.inputs.bind(run_binds, metadata.read_count);
  out.outputs.bind(run_binds, metadata.write_count);
  for (std::size_t local = 0u; local < metadata.binding_accesses.size();
       ++local) {
    const std::uint64_t index = step.graph_binding_indices[local];
    if (index >= source.size) {
      return false;
    }

    const rund::kernel::ComputeBindingAccess access =
        metadata.binding_accesses[local];
    if (access == rund::kernel::ComputeBindingAccess::Read) {
      if (!out.inputs.push(index)) {
        return false;
      }
    } else if (access == rund::kernel::ComputeBindingAccess::Write) {
      if (!out.outputs.push(index)) { return false; }
    } else {
      return false;
    }
  }
  return out.valid() && out.inputs.size() == metadata.read_count &&
         out.outputs.size() == metadata.write_count;
}

}  // namespace rund::node::accel::detail
