#include "local.hpp"

namespace rund_node_test_pipeline {

int CheckReusableCheckpoints(rund::compute::Device &device,
                             const Backend backend) {
  auto prepared = checkpoint::Prepare(device, backend);
  if (!prepared.context) {
    return prepared.error;
  }
  checkpoint::Context &context = *prepared.context;
  if (const int result = checkpoint::CheckInitialAndParity(context);
      result != 0) {
    return result;
  }
  if (const int result = checkpoint::CheckBusyAndCopy(context); result != 0) {
    return result;
  }
  if (const int result = checkpoint::CheckAliasRejection(context);
      result != 0) {
    return result;
  }
  if (const int result = checkpoint::CheckStorageCapacity(context);
      result != 0) {
    return result;
  }
  if (const int result = checkpoint::CheckPortability(context); result != 0) {
    return result;
  }
  return checkpoint::CheckBoundaries(context);
}

} // namespace rund_node_test_pipeline
