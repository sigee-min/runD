#include "local.hpp"

bool ReadyManyBoundaryTimeoutMatches(const ReadyManyBoundaryCase &boundary) {
  return !boundary.negative_timeout.ok() &&
         boundary.negative_timeout.code() ==
             rund::ReasonCode::TimerDurationInvalid &&
         boundary.zero_timeout.ok() && boundary.zero_timeout.events == 0u &&
         boundary.zero_timeout.timed_out() &&
         boundary.zero_timeout.code() == rund::ReasonCode::IoTimedOut &&
         boundary.parked_timeout.ok() &&
         boundary.parked_timeout.events == 0u &&
         boundary.parked_timeout.timed_out() &&
         boundary.parked_timeout.code() == rund::ReasonCode::IoTimedOut;
}
