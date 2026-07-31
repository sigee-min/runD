#include "reject/local.hpp"

bool NetVectoredRejectsInvalidSlicesAndCapacity() {
  VECTORED_ASSERT(NetVectoredRejectsEmptySlices());
  VECTORED_ASSERT(NetVectoredRejectsNullSlices());
  VECTORED_ASSERT(NetVectoredRejectsCapacityOverflow());
  VECTORED_ASSERT(NetVectoredReportsWouldBlock());
  return true;
}
