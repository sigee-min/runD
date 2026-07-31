#include "state.hpp"

#include <memory>
#include <span>

namespace rund::compute::detail {

std::size_t flow_count(const std::shared_ptr<FlowState> &flow) noexcept {
  return flow == nullptr || flow->inputs.empty()
             ? 0u
             : flow->values[flow->inputs.front() - 1u].count;
}

std::size_t flow_output_count(const std::shared_ptr<FlowState> &flow) noexcept {
  return flow == nullptr || flow->output == 0u
             ? 0u
             : flow->values[flow->output - 1u].count;
}

std::size_t flow_value_count(const std::shared_ptr<FlowState> &flow,
                             const std::uint32_t value) noexcept {
  return flow == nullptr || value == 0u || value > flow->values.size()
             ? 0u
             : flow->values[value - 1u].count;
}

FixedFormat flow_value_format(const std::shared_ptr<FlowState> &flow,
                              const std::uint32_t value) noexcept {
  return flow == nullptr || value == 0u || value > flow->values.size()
             ? FixedFormat{}
             : flow->values[value - 1u].fixed_format;
}

std::uint32_t flow_value(const std::shared_ptr<FlowState> &flow) noexcept {
  return flow == nullptr ? 0u : flow->output;
}

std::span<const HostView>
flow_bindings(const std::shared_ptr<FlowState> &flow) noexcept {
  return flow == nullptr ? std::span<const HostView>{}
                         : std::span<const HostView>{flow->bindings};
}

} // namespace rund::compute::detail
