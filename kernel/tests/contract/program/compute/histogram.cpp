#include "histogram/local.hpp"

namespace program_compute_contract {

int RunHistogramContract() {
  if (HistogramReject() != 0) {
    return 1;
  }
  if (HistogramIdentity() != 0) {
    return 1;
  }
  if (HistogramShape() != 0) {
    return 1;
  }
  if (HistogramReference() != 0) {
    return 1;
  }
  return 0;
}

}  // namespace program_compute_contract
