#pragma once

#include <rund/compute.hpp>

namespace rund_node_flow_contract {

[[nodiscard]] int CheckRecordFields(rund::compute::Backend);
[[nodiscard]] int CheckRecordRuntime(rund::compute::Backend);
[[nodiscard]] int CheckRecordSchema(rund::compute::Backend);

} // namespace rund_node_flow_contract
