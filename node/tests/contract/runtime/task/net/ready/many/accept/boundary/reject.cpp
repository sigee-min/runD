#include "local.hpp"

bool ReadyManyBoundaryRejectMatches(const ReadyManyBoundaryCase &boundary) {
  return !boundary.invalid.ok() &&
         boundary.invalid.code() == rund::ReasonCode::IoFdInvalid &&
         !boundary.duplicate.ok() &&
         boundary.duplicate.code() == rund::ReasonCode::TaskInvalid;
}
