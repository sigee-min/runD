#include <rund/storage.hpp>

int main() {
  rund::storage::Budget budget{1024u};
  if (!budget) {
    return budget.exit_code();
  }

  auto reservation = budget.reserve(256u);
  if (!reservation) {
    return reservation.exit_code();
  }
  const rund::storage::Status committed = reservation.commit(
      rund::storage::Usage{.physical_bytes = 128u, .allocated_bytes = 256u});
  if (!committed) {
    return committed.exit_code();
  }

  const rund::storage::Report used = budget.report();
  if (!used) {
    return used.exit_code();
  }
  if (used.allocated_bytes != 256u || used.available_bytes != 768u) {
    return 2;
  }

  const rund::storage::Status refunded = reservation.refund();
  return refunded.exit_code();
}
