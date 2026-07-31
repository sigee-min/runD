#include "local.hpp"

bool NetCancellationManyCountersMatch(const NetCancellationManyCase &many) {
  return many.report.tasks().reactor().waits_canceled() >= 2u &&
         many.report.tasks().reactor().timeout_timer_cancels() >= 1u &&
         many.report.tasks().failed() == 0u;
}
