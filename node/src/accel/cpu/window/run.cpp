#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../kernel/backend/run.hpp"
#include "../../range_aggregate/execution/run.hpp"
#include "../../window/shape.hpp"
#include "../buffer/batch.hpp"
#include <kernel/program/compute/window/reference.hpp>

#include <cstring>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] rund::AccelCheck
ReadWindowCount(const rund::AccelDevice &pick, const BoundControl &bound,
                const rund::kernel::WindowPlan &plan,
                rund::kernel::u64 &count) noexcept {
  count = plan.input_count;
  if (plan.count_source == rund::kernel::ComputeCountSource::Descriptor) {
    return !bound.active() ? rund::AccelCheck{true, "ok"}
                           : rund::AccelCheck{false, "compute_window_invalid"};
  }
  const rund::kernel::GraphControl &control = bound.control;
  const rund::kernel::GraphControlSource expected =
      plan.count_source == rund::kernel::ComputeCountSource::BufferU64
          ? rund::kernel::GraphControlSource::U64
          : rund::kernel::GraphControlSource::U32;
  if (!control.has_count() || control.has_predicate() ||
      control.count_source != expected ||
      control.capacity != plan.input_count || control.iteration != 0u ||
      bound.count == nullptr || bound.count_handle == nullptr) {
    return rund::AccelCheck{false, "compute_window_invalid"};
  }

  CpuBufferResult source{};
  CpuResidentReq request{.ref = bound.count,
                         .handle = bound.count_handle,
                         .usage = rund::kernel::kResidentUsageRead,
                         .out = &source};
  LookupCpuResidentBatch(pick, &request, 1u);
  const std::size_t width =
      expected == rund::kernel::GraphControlSource::U64 ? 8u : 4u;
  std::uint64_t offset = 0u;
  if (!source.check.ok || source.buffer == nullptr ||
      !rund::kernel::checked::add(bound.count->offset_bytes,
                                  control.count_byte_offset, offset) ||
      offset > source.buffer->data.size() ||
      source.buffer->data.size() - static_cast<std::size_t>(offset) < width) {
    return rund::AccelCheck{false, "compute_window_invalid"};
  }
  if (width == 8u) {
    std::memcpy(&count, source.buffer->data.data() + offset, sizeof(count));
  } else {
    rund::kernel::u32 value = 0u;
    std::memcpy(&value, source.buffer->data.data() + offset, sizeof(value));
    count = value;
  }
  return count <= plan.input_count
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_workset_overflow"};
}

[[nodiscard]] rund::kernel::WindowPlan
ActiveWindowPlan(const rund::kernel::WindowPlan &capacity,
                 const rund::kernel::u64 count) noexcept {
  if (capacity.count_source == rund::kernel::ComputeCountSource::Descriptor) {
    return capacity;
  }
  rund::kernel::WindowPlan active = capacity;
  active.input_count = count;
  active.output_count = count;
  active.input_bytes = count * active.element_bytes;
  active.output_bytes = count * active.element_bytes;
  active.pass_count = count == 0u ? 0u : 1u;
  return active;
}

} // namespace

rund::AccelCheck ExecuteCpuWindow(const rund::AccelDevice &pick,
                                  const rund::kernel::WindowDesc &desc,
                                  const rund::kernel::WindowPlan &plan,
                                  const RangePlan &range,
                                  const RangeBinds &bindings,
                                  const BoundControl &control) {
  if (!pick.check.ok || !WindowShapeOk(desc, plan, bindings) ||
      !WindowRangePlanMatches(plan, range) ||
      range.candidate().disposition() != RangePath::Direct ||
      range.temporary_count() != 0u) {
    return rund::AccelCheck{false, "compute_window_invalid"};
  }
  CpuBufferResult input{};
  CpuBufferResult output{};
  CpuResidentReq reqs[] = {
      CpuResidentReq{.ref = bindings.input,
                     .handle = bindings.input_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &input},
      CpuResidentReq{.ref = bindings.output,
                     .handle = bindings.output_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &output},
  };
  LookupCpuResidentBatch(pick, reqs);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!input.check.ok || !output.check.ok || input.buffer == nullptr ||
      output.buffer == nullptr || adapter == nullptr) {
    return rund::AccelCheck{false, "compute_window_invalid"};
  }

  rund::kernel::u64 count = 0u;
  const rund::AccelCheck count_check =
      ReadWindowCount(pick, control, plan, count);
  if (!count_check.ok) {
    return count_check;
  }
  const std::optional<RangeRun> run = RangeRun::make(range, count);
  if (!run.has_value()) {
    return rund::AccelCheck{false, "compute_window_invalid"};
  }
  if (count == 0u) {
    return rund::AccelCheck{true, "ok"};
  }

  const rund::kernel::WindowPlan active = ActiveWindowPlan(plan, count);
  const void *const input_data =
      input.buffer->data.data() + bindings.input->offset_bytes;
  void *const output_data =
      output.buffer->data.data() + bindings.output->offset_bytes;
  rund::kernel::WindowResult result{};
  switch (active.domain) {
  case rund::kernel::ComputeDomain::I32:
    result = rund::kernel::ReferenceWindowI32(
        static_cast<const rund::kernel::i32 *>(input_data),
        static_cast<rund::kernel::i32 *>(output_data), active);
    break;
  case rund::kernel::ComputeDomain::U32:
    result = rund::kernel::ReferenceWindowU32(
        static_cast<const rund::kernel::u32 *>(input_data),
        static_cast<rund::kernel::u32 *>(output_data), active);
    break;
  case rund::kernel::ComputeDomain::I64:
    result = rund::kernel::ReferenceWindowI64(
        static_cast<const rund::kernel::i64 *>(input_data),
        static_cast<rund::kernel::i64 *>(output_data), active);
    break;
  case rund::kernel::ComputeDomain::U64:
    result = rund::kernel::ReferenceWindowU64(
        static_cast<const rund::kernel::u64 *>(input_data),
        static_cast<rund::kernel::u64 *>(output_data), active);
    break;
  case rund::kernel::ComputeDomain::Fixed:
    result = active.element == rund::kernel::WindowElement::U32
                 ? rund::kernel::ReferenceWindowFixedI32(
                       static_cast<const rund::kernel::i32 *>(input_data),
                       static_cast<rund::kernel::i32 *>(output_data), active)
                 : rund::kernel::ReferenceWindowFixedI64(
                       static_cast<const rund::kernel::i64 *>(input_data),
                       static_cast<rund::kernel::i64 *>(output_data), active);
    break;
  }
  if (!result.ok) {
    return rund::AccelCheck{false, result.reason};
  }
  std::uint64_t dispatches = 0u;
  for (std::size_t index = 0u; index < range.stage_count(); ++index) {
    const std::optional<RangeDispatch> stage = run->stage(index);
    if (!stage.has_value()) {
      return rund::AccelCheck{false, "compute_window_invalid"};
    }
    dispatches += static_cast<std::uint64_t>(stage->active());
  }
  RecordCpuDispatches(*adapter, dispatches);
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail
