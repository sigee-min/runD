#pragma once

namespace rund::compute::graph {
struct Info;
}

namespace package_compute::graph_services {

[[nodiscard]] int CheckSessionCompile();
[[nodiscard]] int CheckResourcePlan();
[[nodiscard]] bool ValidateGraph(const rund::compute::graph::Info &graph);
[[nodiscard]] int CheckExecution();

} // namespace package_compute::graph_services
