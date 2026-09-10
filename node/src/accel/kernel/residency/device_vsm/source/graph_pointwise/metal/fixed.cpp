#include "internal.hpp"

namespace rund::node::accel::detail::device_vsm_graph_pointwise::metal {

bool append_fixed_helpers(
    std::string &source,
    const std::span<const DeviceVsmGraphPointwiseStage> stages,
    const rund::kernel::ArtifactKey &key) {
  namespace lowering = rund::kernel::compute_lowering_detail;
  lowering::ParsedIR helpers{};
  helpers.scalar_mode = lowering::DomainModeFor(key.scalar, key.domain);
  helpers.ok = true;
  for (const DeviceVsmGraphPointwiseStage &stage : stages) {
    if (stage.artifact == nullptr || stage.input == nullptr ||
        !stage.input->parsed.ok || stage.artifact->key.api != key.api ||
        stage.artifact->key.scalar != key.scalar ||
        stage.artifact->key.domain != key.domain ||
        stage.input->parsed.scalar_mode !=
            lowering::DomainModeFor(key.scalar, key.domain)) {
      return false;
    }
    helpers.nodes.insert(helpers.nodes.end(), stage.input->parsed.nodes.begin(),
                         stage.input->parsed.nodes.end());
  }
  lowering::AppendMetalFixedOpHelpers(source, helpers, key);
  return true;
}

} // namespace rund::node::accel::detail::device_vsm_graph_pointwise::metal
