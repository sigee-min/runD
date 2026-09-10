#include "model.hpp"

#include "source/classify.hpp"
#include "source/prefix.hpp"
#include "source/prelude.hpp"
#include "source/reduce.hpp"
#include "source/scatter.hpp"

#include "../../../domain.hpp"
#include "../../../kernel/backend/source/storage.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] const char *OpName(const rund::kernel::ReduceOp op) noexcept {
  switch (op) {
  case rund::kernel::ReduceOp::Sum:
    return "sum";
  case rund::kernel::ReduceOp::CountNonzero:
    return "count";
  case rund::kernel::ReduceOp::Min:
    return "min";
  case rund::kernel::ReduceOp::Max:
    return "max";
  }
  return "invalid";
}

template <typename Sink>
[[nodiscard]] bool EmitMetalSegmentedReduceSource(
    Sink &source, const rund::kernel::ReduceOp op,
    const rund::kernel::ComputeDomain domain) noexcept(
    noexcept(source += std::string_view{})) {
  return EmitMetalSegmentedReducePreludeSource(source) &&
         EmitMetalSegmentedReduceClassifySource(source) &&
         EmitMetalSegmentedReducePrefixSource(source) &&
         EmitMetalSegmentedReduceScatterSource(source) &&
         EmitMetalSegmentedReduceReduceSource(source, op, domain);
}

} // namespace

std::string
MetalSegmentedReduceSource(const rund::kernel::ReduceOp op,
                           const rund::kernel::ComputeDomain domain) {
  const auto emit = [op, domain](auto &sink) noexcept(noexcept(
      EmitMetalSegmentedReduceSource(sink, op, domain))) {
    return EmitMetalSegmentedReduceSource(sink, op, domain);
  };
  return backend_source_recipe::materialize(emit);
}

bool MetalSegmentedReduceSourceUpperBytes(
    const rund::kernel::ReduceOp op,
    const rund::kernel::ComputeDomain domain,
    std::uint64_t &upper) noexcept {
  const auto emit = [op, domain](
                        backend_source_recipe::CountSink &sink) noexcept {
    return EmitMetalSegmentedReduceSource(sink, op, domain);
  };
  return backend_source_recipe::bytes(emit, upper);
}

std::string
MetalSegmentedReduceKey(const rund::kernel::SegmentedReducePlan &plan,
                        const rund::kernel::ComputeDomain domain) {
  std::string key = "segmented-reduce.";
  key += OpName(plan.op);
  key += IsSignedDomain(domain) ? ".i" : ".u";
  key += plan.element_bytes == sizeof(rund::kernel::u64) ? "64" : "32";
  return key;
}

std::string
MetalSegmentedReduceName(const rund::kernel::SegmentedReducePlan &plan,
                         const rund::kernel::ComputeDomain domain) {
  std::string name = "rund_compute_segmented_reduce_";
  name += IsSignedDomain(domain) ? "i" : "u";
  name += plan.element_bytes == sizeof(rund::kernel::u64) ? "64" : "32";
  return name;
}

#endif

} // namespace rund::node::accel::detail
