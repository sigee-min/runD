#include "local.hpp"

namespace rund_node_flow_contract {

int CheckRecords(const rund::compute::Backend backend) {
  // Keep the original fail-fast sequence: typed field arithmetic precedes
  // record projection/execution, then schema selection and bounded output.
  if (const int result = CheckRecordFields(backend); result != 0) {
    return result;
  }
  if (const int result = CheckRecordRuntime(backend); result != 0) {
    return result;
  }
  return CheckRecordSchema(backend);
}

} // namespace rund_node_flow_contract
