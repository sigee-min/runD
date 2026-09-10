#include "contract/program/compute/lowering/support.hpp"
#include "local.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/resource.hpp>

#include <string_view>
#include <vector>

namespace program_compute_metadata_contract {
namespace {

using namespace program_compute_contract::lowering_support;

struct MixedResourceBody final {
  using Mode = rund::compute_dsl::detail::ScalarMode;
  using Access = rund::compute_dsl::detail::BindingKind;
  using Binding = rund::compute_dsl::detail::BindingRuntime;

  [[nodiscard]] static constexpr Mode scalar_mode() noexcept {
    return Mode::I32;
  }
  [[nodiscard]] constexpr rund::kernel::ComputeFixedFormat
  fixed_format() const noexcept {
    return {};
  }
  [[nodiscard]] constexpr bool ok() const noexcept { return true; }
  [[nodiscard]] constexpr const char *reason() const noexcept { return "ok"; }
  [[nodiscard]] const std::vector<Binding> &bindings() const noexcept {
    return values;
  }

  std::vector<Binding> values{
      Binding{.kind = Access::Read,
              .numeric_mode = Mode::U32,
              .name = "index",
              .element_bytes = 4u},
      Binding{.kind = Access::Read,
              .numeric_mode = Mode::I32,
              .name = "direct",
              .element_bytes = 4u},
      Binding{.kind = Access::Read,
              .numeric_mode = Mode::I32,
              .name = "uniform",
              .element_bytes = 4u},
      Binding{.kind = Access::Read,
              .numeric_mode = Mode::I32,
              .name = "indexed",
              .element_bytes = 4u},
      Binding{.kind = Access::Write,
              .numeric_mode = Mode::I32,
              .name = "output",
              .element_bytes = 4u},
  };
};

[[nodiscard]] rund::kernel::ComputeIR BuildMixedResourceIr() {
  using namespace rund::compute_dsl::detail;
  MixedResourceBody body{};
  BuildContext context{body.bindings(), MixedResourceBody::scalar_mode()};
  const Expr direct = DynamicRead(context, 1u);
  const Expr uniform = DynamicUniformRead(context, 2u);
  const Expr indexed = DynamicReadAt(context, 3u, 0u, 4u);
  const Expr first = Binary(rund::kernel::IrOp::Add, direct, uniform);
  const Expr second = Binary(rund::kernel::IrOp::Add, first, indexed);
  DynamicWrite(context, 4u, second);
  return BuildIr("resource-summary", body, context);
}

int CheckFixedLane32Metadata() {
  const auto op = BuildFixedLane32Op(7);
  const rund::kernel::ExecutionMetadata metadata =
      rund::kernel::BuildExecutionMetadata(op.ir(),
                                           rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(metadata.ok);
  TEST_ASSERT(std::string_view{metadata.reason} == "ok");
  TEST_ASSERT(metadata.map.op_hash_hi == op.ir().op_hash_hi);
  TEST_ASSERT(metadata.map.op_hash_lo == op.ir().op_hash_lo);
  TEST_ASSERT(metadata.map.api == rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(metadata.map.scalar == rund::kernel::ComputeScalar::Lane32);
  TEST_ASSERT(metadata.map.input_buffer_count == 2u);
  TEST_ASSERT(metadata.map.input_bytes_per_tile == 8u);
  TEST_ASSERT(metadata.map.output_bytes_per_tile == 4u);
  TEST_ASSERT(metadata.map.param_bytes == 4u);
  TEST_ASSERT(metadata.map.metadata_bytes_per_tile == 4u);
  TEST_ASSERT(metadata.read_count == 2u);
  TEST_ASSERT(metadata.write_count == 1u);
  TEST_ASSERT(metadata.direct_read_mask == 0x3u);
  TEST_ASSERT(metadata.uniform_read_mask == 0u);
  TEST_ASSERT(metadata.read_routes.empty());
  TEST_ASSERT(metadata.resource_summary.ok);
  TEST_ASSERT(metadata.resource_summary.analysis_version ==
              rund::kernel::kComputeResourceAnalysisVersion);
  TEST_ASSERT(metadata.resource_summary.peak_live_words == 16u);
  TEST_ASSERT(metadata.resource_summary.direct_read_count == 2u);
  TEST_ASSERT(metadata.resource_summary.uniform_read_count == 0u);
  TEST_ASSERT(metadata.resource_summary.indexed_read_count == 0u);
  TEST_ASSERT(metadata.resource_summary.write_count == 1u);
  TEST_ASSERT(metadata.binding_accesses.size() == 3u);
  TEST_ASSERT(metadata.binding_accesses[0] ==
              rund::kernel::ComputeBindingAccess::Read);
  TEST_ASSERT(metadata.binding_accesses[1] ==
              rund::kernel::ComputeBindingAccess::Read);
  TEST_ASSERT(metadata.binding_accesses[2] ==
              rund::kernel::ComputeBindingAccess::Write);
  TEST_ASSERT(metadata.binding_names.size() == 3u);
  TEST_ASSERT(metadata.binding_names[0] == "pos");
  TEST_ASSERT(metadata.binding_names[1] == "vel");
  TEST_ASSERT(metadata.binding_names[2] == "out");
  TEST_ASSERT(metadata.input_element_bytes.size() == 2u);
  TEST_ASSERT(metadata.input_element_bytes[0] == 4u);
  TEST_ASSERT(metadata.input_element_bytes[1] == 4u);
  TEST_ASSERT(metadata.param_storage.size() == 4u);
  TEST_ASSERT(metadata.param_storage[0] == 7u);
  TEST_ASSERT(metadata.param_storage[1] == 0u);
  TEST_ASSERT(metadata.param_storage[2] == 0u);
  TEST_ASSERT(metadata.param_storage[3] == 0u);
  return 0;
}

int CheckResourceSummary() {
  const rund::kernel::ComputeIR ir = BuildMixedResourceIr();
  const rund::kernel::ComputeResourceSummary summary =
      rund::kernel::BuildComputeResourceSummary(
          ir, rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(ir.ok);
  TEST_ASSERT(summary.ok);
  TEST_ASSERT(std::string_view{summary.reason} == "ok");
  TEST_ASSERT(summary.analysis_version ==
              rund::kernel::kComputeResourceAnalysisVersion);
  TEST_ASSERT(summary.peak_live_words == 4u);
  TEST_ASSERT(summary.direct_read_count == 1u);
  TEST_ASSERT(summary.uniform_read_count == 1u);
  TEST_ASSERT(summary.indexed_read_count == 1u);
  TEST_ASSERT(summary.write_count == 1u);
  return 0;
}

} // namespace

int CheckMetadataSurface() {
  if (const int result = CheckFixedLane32Metadata(); result != 0) {
    return result;
  }
  return CheckResourceSummary();
}

} // namespace program_compute_metadata_contract
