#include "test/assert.hpp"

#include "io/local.hpp"

int RunRuntimeTaskNetFrameIoContract() {
  TEST_ASSERT(FrameIoSuccessfulWriteReadPreservesBytes());
  TEST_ASSERT(FrameIoPayloadTooLargeRejectsBeforeWrite());
  TEST_ASSERT(FrameIoVectoredCapacityRejectsDirectly());
  TEST_ASSERT(FrameIoDestinationBufferTooSmallRejectsAfterHeader());
  TEST_ASSERT(FrameIoZeroLengthPayloadSucceedsWithNoPayloadBytes());
  TEST_ASSERT(FrameIoBudgetExhaustionReportsIncomplete());
  return 0;
}
