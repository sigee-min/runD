#pragma once

#include <rund/compute.hpp>

namespace rund_node_test_flow_primitives {

struct Left final {};
struct Right final {};
struct Sum final {};
struct Values final {};

[[nodiscard]] rund::compute::FlowBuilder Target();

[[nodiscard]] int CheckOutputs();
[[nodiscard]] int CheckIdentityProjection();
[[nodiscard]] int CheckComposition();
[[nodiscard]] int CheckBoundedPipe();
[[nodiscard]] int CheckBoundedGather();
[[nodiscard]] int CheckIndexedMap();
[[nodiscard]] int CheckScatterReduce();
[[nodiscard]] int CheckBoundedCollectiveRepeat();
[[nodiscard]] int CheckBoundedBoundaryConsumers();
[[nodiscard]] int CheckGroup();
[[nodiscard]] int CheckJoin();
[[nodiscard]] int CheckGroupCapacity();
[[nodiscard]] int CheckPoolSurface();

} // namespace rund_node_test_flow_primitives
