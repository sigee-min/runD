#include "cases.hpp"

namespace {

int Run(int (*test)()) {
  const int result = test();
  return result == 0 ? 0 : result;
}

} // namespace

int RunReductionFoldContract() {
  if (const int result = Run(RunFoldSlotsContract); result != 0) {
    return result;
  }
  if (const int result = Run(RunFoldHashContract); result != 0) {
    return result;
  }
  if (const int result = Run(RunFoldGraphContract); result != 0) {
    return result;
  }
  if (const int result = Run(RunStrictFoldContract); result != 0) {
    return result;
  }
  return 0;
}
