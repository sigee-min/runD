#include "local.hpp"

namespace rund::compute::detail::virtual_run_overlap::prepare_detail {

bool is_accelerator(const Context &context) noexcept {
  return context.backend == Backend::Accelerator;
}

bool release_prefetch_aliases(Context &context,
                              const bool source_known) noexcept {
  if (!is_accelerator(context)) {
    return true;
  }
  return context.pool->prefetch[context.lane].release_aliases(
      context.pool->authority(), source_known);
}

bool cancel_prefetch_receipt(Context &context,
                             residency::PrefetchReceipt &receipt,
                             const bool source_known) noexcept {
  if (!is_accelerator(context)) {
    return true;
  }
  return context.pool->prefetch[context.lane].cancel(
      context.pool->authority(), receipt, source_known, true);
}

bool wait_prefetch(residency::Pool &pool, std::array<bool, 2u> &pending,
                   const bool publish, const bool source_known,
                   const Backend backend) noexcept {
  const bool use_accelerator = backend == Backend::Accelerator;
  bool clean = true;
  for (std::size_t lane = 0u; lane < pending.size(); ++lane) {
    if (!pending[lane]) {
      continue;
    }
    if (!publish) {
      const bool cancelled =
          use_accelerator
              ? pool.prefetch[lane].cancel(pool.authority(), source_known, true)
              : true;
      clean = cancelled && clean;
      if (cancelled) {
        pending[lane] = false;
      }
      continue;
    }
    residency::PrefetchReceipt receipt = pool.prefetch[lane].wait();
    bool terminal = true;
    if (receipt.token != 0u) {
      if (!receipt.status) {
        terminal = pool.prefetch[lane].cancel(pool.authority(), receipt,
                                              source_known, true);
      } else if (!pool.authority().complete(receipt.token, true, false)) {
        terminal = pool.prefetch[lane].cancel(pool.authority(), receipt,
                                              source_known, true);
      } else {
        terminal =
            pool.prefetch[lane].release_aliases(pool.authority(), source_known);
      }
    } else {
      terminal =
          pool.prefetch[lane].release_aliases(pool.authority(), source_known);
    }
    clean = terminal && clean;
    if (terminal) {
      pending[lane] = false;
    }
  }
  return clean;
}

} // namespace rund::compute::detail::virtual_run_overlap::prepare_detail
