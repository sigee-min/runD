#include "local.hpp"

int RunComputeFlowPrimitivesContract() {
  using namespace rund_node_test_flow_primitives;
  if (const int result = CheckOutputs(); result != 0) {
    return 10 + result;
  }
  if (const int result = CheckIdentityProjection(); result != 0) {
    return 20 + result;
  }
  if (const int result = CheckComposition(); result != 0) {
    return 30 + result;
  }
  if (const int result = CheckBoundedPipe(); result != 0) {
    return 40 + result;
  }
  if (const int result = CheckBoundedGather(); result != 0) {
    return 45 + result;
  }
  if (const int result = CheckIndexedMap(); result != 0) {
    return 47 + result;
  }
  if (const int result = CheckScatterReduce(); result != 0) {
    return 48 + result;
  }
  if (const int result = CheckBoundedCollectiveRepeat(); result != 0) {
    return 52 + result;
  }
  if (const int result = CheckBoundedBoundaryConsumers(); result != 0) {
    return 60 + result;
  }
  if (const int result = CheckGroup(); result != 0) {
    return 70 + result;
  }
  if (const int result = CheckJoin(); result != 0) {
    return 80 + result;
  }
  if (const int result = CheckGroupCapacity(); result != 0) {
    return 90 + result;
  }
  if (const int result = CheckPoolSurface(); result != 0) {
    return 100 + result;
  }
  return 0;
}
