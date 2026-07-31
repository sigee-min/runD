#include "algebra/local.hpp"

namespace program_compute_contract {

int RunNumericAlgebraContract() {
  using namespace numeric_algebra_contract;
  if (RunMatrix() != 0 || RunFactor() != 0 || RunSolve() != 0 ||
      RunSpectrum() != 0 || RunScratch() != 0 || RunAccuracy() != 0) {
    return 1;
  }
  return RunIdentity();
}

} // namespace program_compute_contract
