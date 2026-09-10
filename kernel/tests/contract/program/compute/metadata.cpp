#include "metadata/local.hpp"

namespace program_compute_contract {

int RunComputeMetadataContract() {
  using namespace program_compute_metadata_contract;
  if (const int result = CheckResourceContracts(); result != 0) {
    return 1;
  }
  if (const int result = CheckMetadataSurface(); result != 0) {
    return 1;
  }
  if (const int result = CheckMetadataRejections(); result != 0) {
    return 1;
  }
  if (const int result = CheckInputIdentity(); result != 0) {
    return 1;
  }
  if (const int result = CheckRetention(); result != 0) {
    return 1;
  }
  return 0;
}

} // namespace program_compute_contract
