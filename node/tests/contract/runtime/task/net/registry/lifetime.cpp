#include "test/assert.hpp"

#include "lifetime/local.hpp"

int RunRuntimeTaskNetRegistryLifetimeContract() {
  TEST_ASSERT(RunNetRegistryLifetimeGenerationCase() == 0);
  TEST_ASSERT(RunNetRegistryLifetimeReplacementCase() == 0);
  TEST_ASSERT(RunNetRegistryLifetimeStabilityCase() == 0);
  return 0;
}
