#include "local.hpp"

#include "../../target/selection.hpp"

#include <array>
#include <cstdint>

namespace rund_node_bounded_contract {

[[nodiscard]] int CheckBackends() {
  using namespace rund::compute;
  std::uint64_t group_graph = 0u;
  std::uint64_t group_many = 0u;
  std::uint64_t group_fewer = 0u;
  std::uint64_t expand_graph = 0u;
  std::uint64_t expand_output = 0u;
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (backend != Backend::Cpu) {
      const int filter = CheckAcceleratorFilterLaws(backend);
      if (filter != 0) {
        return 20 + 10 * static_cast<int>(backend) + filter;
      }
    }
    Stats expand_stats{};
    if (const int expand = CheckExpandBackend(backend, &expand_stats);
        expand != 0) {
      return expand;
    }
    if (expand_graph == 0u) {
      expand_graph = expand_stats.graph_hash;
      expand_output = expand_stats.output_hash;
    } else if (expand_stats.graph_hash != expand_graph ||
               expand_stats.output_hash != expand_output) {
      return 400 + 10 * static_cast<int>(backend);
    }
    const int rewrite =
        CheckGroupRewriteBackend(backend, group_graph, group_many, group_fewer);
    if (rewrite != 0) {
      return 40 + 10 * static_cast<int>(backend) + rewrite;
    }
    if (const int invalid = CheckInvalidBoundedCount(backend); invalid != 0) {
      return 80 + 10 * static_cast<int>(backend) + invalid;
    }
    if (const int compact = CheckCompactCapacityBackend(backend);
        compact != 0) {
      return 120 + 20 * static_cast<int>(backend) + compact;
    }
  }
  if (rund::node::test_contract::backend_selected(Backend::Vulkan)) {
    if (const int widths = CheckVulkanPartitionPipelineWidths(); widths != 0) {
      return 220 + widths;
    }
  }
  return 0;
}

} // namespace rund_node_bounded_contract
