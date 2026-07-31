#include "local.hpp"

bool NetStatsCancelCountersMatch(const NetStatsCancelCase &stats) {
  return stats.report.tasks().reactor().timed_waits_registered() >= 1u &&
         stats.report.tasks().reactor().timeout_timer_cancels() >= 1u &&
         stats.report.tasks().reactor().close_invalidated_waits() >= 1u &&
         stats.report.tasks().reactor().waits_canceled() >= 1u;
}
