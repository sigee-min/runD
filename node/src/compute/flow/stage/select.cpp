#include "../recipe.hpp"

#include <algorithm>
#include <memory>
#include <span>

namespace rund::compute::detail {

void flow_pick(const std::shared_ptr<FlowState> &flow,
               const std::uint32_t value) {
  if (flow == nullptr || !flow->status) {
    return;
  }
  if (value == 0u || value > flow->values.size()) {
    reject(*flow, Reason::FlowValueInvalid);
    return;
  }
  flow->output = value;
  flow->outputs.clear();
  flow->logical_outputs.clear();
}

void flow_outputs(const std::shared_ptr<FlowState> &flow,
                  const std::span<const std::uint32_t> values) {
  if (flow == nullptr || !flow->status) {
    return;
  }
  if (values.empty() || values.size() > MaxOutputs) {
    reject(*flow, Reason::GraphOutputCapacity);
    return;
  }
  for (const std::uint32_t value : values) {
    if (value == 0u || value > flow->values.size()) {
      reject(*flow, Reason::FlowValueInvalid);
      return;
    }
  }
  try {
    flow->outputs.clear();
    flow->logical_outputs.clear();
    flow->outputs.reserve(values.size());
    for (const std::uint32_t value : values) {
      if (std::find(flow->outputs.begin(), flow->outputs.end(), value) ==
          flow->outputs.end()) {
        flow->outputs.push_back(value);
      }
    }
    if (flow->outputs.size() != values.size()) {
      flow->logical_outputs.assign(values.begin(), values.end());
    }
  } catch (const std::bad_alloc &) {
    reject(*flow, Reason::FlowCapacity);
  }
}

} // namespace rund::compute::detail
