#include "local.hpp"

bool NetCancellationManyCancelMatches(const NetCancellationManyCase &many) {
  return many.source_valid && many.token_valid && many.scope.ok() &&
         many.cancel_yield.ok() && many.cancel_ok && !many.ready.ok() &&
         !many.ready.timed_out() && many.ready.events == 0u &&
         many.ready.code() == rund::ReasonCode::TaskCancelled;
}
