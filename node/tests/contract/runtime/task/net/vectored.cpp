#include "test/assert.hpp"

#include "vectored/local.hpp"

int RunRuntimeTaskNetVectoredContract() {
  TEST_ASSERT(NetVectoredSendRecvPreservesSliceOrder());
  TEST_ASSERT(NetVectoredPartialCompletionHashesCompletedPrefix());
  TEST_ASSERT(NetVectoredRejectsInvalidSlicesAndCapacity());
  TEST_ASSERT(NetVectoredRejectsImpossibleSliceMetadata());
  return 0;
}
