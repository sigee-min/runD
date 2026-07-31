#include "local.hpp"

bool NetStatsCloseResultMatches(const NetStatsCancelCase &stats) {
  return stats.close_yielded.ok() && stats.close_result.ok() &&
         !stats.close_wait_result.ok() &&
         stats.close_wait_result.code() == rund::ReasonCode::IoFdInvalid;
}
