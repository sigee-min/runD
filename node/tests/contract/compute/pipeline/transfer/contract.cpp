#include "local.hpp"

#include <cstdio>

int RunComputePipelineTransferContract() {
  int result = 0;
  if (!rund_node_test_pipeline_transfer::CheckBatchSuccessAndAccounting()) {
    result = 1;
  } else if (!rund_node_test_pipeline_transfer::CheckBatchClaimAndFailure()) {
    result = 2;
  } else if (!rund_node_test_pipeline_transfer::CheckBatchDownloadAndHash()) {
    result = 3;
  }
  if (result != 0) {
    std::fprintf(stderr, "pipeline batch transfer result=%d\n", result);
  }
  return result;
}
