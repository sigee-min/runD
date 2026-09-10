#include "model.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace rund_node_flow_contract {

[[nodiscard]] int
CheckBackendContracts(const std::span<const rund::compute::Backend> backends) {
  if (CheckIndexedCapacity() != 0) {
    return 39;
  }
  std::uint64_t expression_graph_hash = 0u;
  std::uint64_t expression_output_hash = 0u;
  std::array<FlowHash, 6u> composition_reference{};
  std::array<FlowHash, 3u> typed_reference{};
  for (const rund::compute::Backend backend : backends) {
    if (const int result = CheckIndexedMap(backend); result != 0) {
      return 40 + result;
    }
    if (const int result = CheckWideMap(backend); result != 0) {
      return 45 + result;
    }
    if (const int result = CheckIndexedAlias(backend); result != 0) {
      return 48 + result;
    }
    if (const int result = CheckFusionBoundaries(backend); result != 0) {
      return 52 + result;
    }
    if (const int result = CheckResetProjection(backend); result != 0) {
      return 56 + result;
    }
    if (const int result = CheckExpressions(backend, expression_graph_hash,
                                            expression_output_hash);
        result != 0) {
      return 50 + result;
    }
    if (const int result = CheckRecords(backend); result != 0) {
      return 60 + result;
    }
    if (!CheckComposition(backend, composition_reference)) {
      return 70 + static_cast<int>(backend);
    }
    if (!CheckTyped(backend, typed_reference)) {
      return 80 + static_cast<int>(backend);
    }
  }
  return 0;
}

} // namespace rund_node_flow_contract
