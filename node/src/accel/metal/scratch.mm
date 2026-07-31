#include "scratch.hpp"

#include "buffer/owner.hpp"
#include "resident.hpp"

#include <new>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

thread_local MetalScratch *active_scratch{};

} // namespace

MetalScratch::MetalScratch(const rund::AccelDevice &pick,
                           const KernelScratchLayout &layout,
                           const RunBinds &binds) noexcept {
  if (!MetalPickOwnsAdapter(pick) || !ValidKernelScratch(layout, binds)) {
    return;
  }
  adapter_ = MetalAdapterFromPick(pick);
  if (adapter_ == nullptr || adapter_->caps.storage_alignment == 0u) {
    return;
  }
  try {
    pages_.reserve(layout.size());
    for (const KernelScratchPage page : layout) {
      const rund::kernel::ResidentBufferRef &ref = binds.refs()[page.slot];
      MetalResidentBufferResult resolved = LookupMetalResidentBuffer(
          pick, ref, binds.handles()[page.slot]);
      if (!resolved.check.ok || resolved.device_buffer == nullptr ||
          ref.offset_bytes > ref.bytes ||
          page.bytes > ref.bytes - ref.offset_bytes) {
        pages_.clear();
        return;
      }
      pages_.push_back(Page{
          .buffer = std::move(resolved.device_buffer),
          .base = ref.offset_bytes,
          .bytes = page.bytes,
      });
    }
  } catch (const std::bad_alloc &) {
    pages_.clear();
    return;
  }
  valid_ = pages_.size() == layout.size();
}

bool MetalScratch::valid() const noexcept { return valid_; }

bool MetalScratch::used() const noexcept { return used_; }

bool MetalScratch::active() const noexcept {
  return scratch::active(pages_);
}

void MetalScratch::reset() noexcept { scratch::reset(pages_); }

MetalRuntimeBuffer MetalScratch::acquire(const std::uint64_t bytes) noexcept {
  if (!valid_ || bytes == 0u || adapter_ == nullptr) {
    return {};
  }
  const scratch::Placement placed =
      scratch::fit(pages_, adapter_->caps.storage_alignment, bytes);
  if (!placed.ok || placed.page >= pages_.size()) {
    return {};
  }
  Page &page = pages_[placed.page];
  used_ = true;
  return MetalRuntimeBuffer{
      .bytes = bytes,
      .usage = MetalBufferUsage::Scratch,
      .buffer = page.buffer,
      .offset = page.base + placed.offset,
      .reused = true,
      .borrowed = true,
  };
}

MetalScratchScope::MetalScratchScope(MetalScratch *const scratch) noexcept
    : prior_{active_scratch} {
  active_scratch = scratch;
}

MetalScratchScope::~MetalScratchScope() { active_scratch = prior_; }

MetalScratch *ActiveMetalScratch() noexcept { return active_scratch; }

#endif

} // namespace rund::node::accel::detail
