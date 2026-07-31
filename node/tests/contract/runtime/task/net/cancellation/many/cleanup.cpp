#include "local.hpp"

bool NetCancellationManyCleanupMatches(const NetCancellationManyCase &many) {
  return many.close.ok() && many.post_close_scope.ok() &&
         many.post_close_sleep.ok();
}
