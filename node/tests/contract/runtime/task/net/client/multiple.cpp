#include "test/assert.hpp"

#include "multiple/local.hpp"

int RunRuntimeTaskNetMultiClientSubstrateContract() {
  TEST_ASSERT(RunNetMultiClientSubstrateCase() == 0);
  return 0;
}
