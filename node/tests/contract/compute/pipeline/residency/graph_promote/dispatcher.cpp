#include "local.hpp"

namespace rund_node_test_pipeline_residency {

[[nodiscard]] int CheckGraphPromote() {
  if (const int wide = graph_promote::CheckWide(); wide != 0) {
    return 50 + wide;
  }
  if (const int grouped = graph_promote::CheckGroup(); grouped != 0) {
    return 100 + grouped;
  }
  return graph_promote::CheckSingle();
}

} // namespace rund_node_test_pipeline_residency
