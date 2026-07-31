#include "local.hpp"

bool ReadyManyBoundaryEmptyMatches(const ReadyManyBoundaryCase &boundary) {
  return boundary.empty.ok() && boundary.empty.events == 0u &&
         !boundary.empty.timed_out() && !boundary.empty_output.ok() &&
         boundary.empty_output.code() == rund::ReasonCode::TaskInvalid;
}
