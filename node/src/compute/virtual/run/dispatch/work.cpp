#include "local.hpp"

namespace rund::compute::detail {

Status prepare_work(const VirtualRunProjection &run,
                    VirtualRunWork &work) noexcept {
  if (work.prepared) {
    return work.prepared_status;
  }
  work.prepared = true;
  if (run.reduction) {
    work.prepared_status = begin_virtual_reduction(run, work.reduction);
    if (!work.prepared_status) {
      return work.prepared_status;
    }
  }
  if (run.scan) {
    work.prepared_status = begin_virtual_scan(run, work.scan);
    if (!work.prepared_status) {
      return work.prepared_status;
    }
  }
  work.prepared_status = Status::success();
  return work.prepared_status;
}

} // namespace rund::compute::detail
