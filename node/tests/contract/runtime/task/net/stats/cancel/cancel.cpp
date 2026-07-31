#include "local.hpp"

bool NetStatsCancelResultMatches(const NetStatsCancelCase &stats) {
  return stats.source_valid && stats.token_valid && stats.cancel_yielded.ok() &&
         stats.cancel_ok && !stats.cancel_result.ok() &&
         stats.cancel_result.code() == rund::ReasonCode::TaskCancelled;
}
