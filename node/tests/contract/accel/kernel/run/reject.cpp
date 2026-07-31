#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/visibility.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>
#include <accel/kernel/value.hpp>

#include <accel/graph/factory/buffer/read.hpp>
#include <accel/graph/factory/buffer/write.hpp>
#include <accel/graph/factory/map.hpp>

#include "local.hpp"
#include <node/accel/context.hpp>

#include <cstdio>
#include <memory>
#include <string_view>
#include <utility>

namespace node_accel_contract::kernel_case {
namespace {

struct IndexedBody final {
  [[nodiscard]] static constexpr rund::compute_dsl::detail::ScalarMode
  scalar_mode() noexcept {
    return rund::compute_dsl::detail::ScalarMode::I64;
  }
  [[nodiscard]] const std::vector<rund::compute_dsl::detail::BindingRuntime> &
  bindings() const noexcept {
    return values;
  }
  [[nodiscard]] constexpr rund::kernel::u64 tile_count() const noexcept {
    return 4u;
  }
  [[nodiscard]] constexpr rund::kernel::ComputeFixedFormat
  fixed_format() const noexcept {
    return {};
  }
  [[nodiscard]] constexpr bool ok() const noexcept { return true; }
  [[nodiscard]] constexpr const char *reason() const noexcept { return "ok"; }

  std::vector<rund::compute_dsl::detail::BindingRuntime> values{
      {.kind = rund::compute_dsl::detail::BindingKind::Read,
       .numeric_mode = rund::compute_dsl::detail::ScalarMode::I64,
       .name = "source",
       .element_bytes = 8u},
      {.kind = rund::compute_dsl::detail::BindingKind::Read,
       .numeric_mode = rund::compute_dsl::detail::ScalarMode::U32,
       .name = "index",
       .element_bytes = 4u},
      {.kind = rund::compute_dsl::detail::BindingKind::Write,
       .numeric_mode = rund::compute_dsl::detail::ScalarMode::I64,
       .name = "output",
       .element_bytes = 8u},
  };
};

[[nodiscard]] rund::compute_dsl::ComputeOp BuildIndexedOp() {
  using namespace rund::compute_dsl::detail;
  IndexedBody body{};
  BuildContext context{body.bindings(), ScalarMode::I64};
  const Expr value = DynamicReadAt(context, 0u, 1u, 6u);
  DynamicWrite(context, 2u, value);
  rund::kernel::ComputeIR ir = BuildIr("indexed-map", body, context);
  const rund::kernel::ComputeMap map = BuildMap(ir, body);
  return rund::compute_dsl::ComputeOp{std::move(ir), map,
                                      std::move(body.values), 4u};
}

} // namespace

[[nodiscard]] bool
ResidentRunRejectsForgedAndForeign(const ResidentRunFixture &fixture) {
  const std::array<rund::AccelRunBinding, 2u> bindings = Bindings(fixture);
  rund::AccelKernel forged_kernel = fixture.kernel;
  forged_kernel.owner =
      std::shared_ptr<void>(fixture.kernel.owner.get(), [](void *) {});
  const rund::AccelEvidence forged_evidence = rund::node::accel::RunAccelKernel(
      fixture.context, forged_kernel,
      RunRequest(bindings, fixture.host_input.size(), true));
  if (!EvidenceReason(forged_evidence, "accel_kernel_run_invalid") ||
      forged_evidence.graph_id_hi != 0u || forged_evidence.graph_id_lo != 0u ||
      forged_evidence.kernel_id != 0u ||
      forged_evidence.backend != rund::AccelApi::Auto) {
    return false;
  }

  int alias_value{};
  forged_kernel = fixture.kernel;
  forged_kernel.owner =
      std::shared_ptr<void>(fixture.kernel.owner, &alias_value);
  if (!EvidenceReason(
          rund::node::accel::RunAccelKernel(
              fixture.context, forged_kernel,
              RunRequest(bindings, fixture.host_input.size(), true)),
          "accel_kernel_run_invalid")) {
    return false;
  }

  const rund::AccelContext other =
      rund::node::accel::OpenAccel(fixture.context.pick);
  if (!other.check.ok) {
    return false;
  }
  if (!EvidenceReason(
          rund::node::accel::RunAccelKernel(
              other, fixture.kernel,
              RunRequest(bindings, fixture.host_input.size(), true)),
          "accel_kernel_run_invalid")) {
    return false;
  }

  const rund::AccelBuffer wrong_shape = rund::node::accel::CreateAccelBuffer(
      fixture.context, BufferDesc(rund::BufferUsage::WriteOnly, 7u));
  if (!wrong_shape.check.ok) {
    return false;
  }
  std::array<rund::AccelRunBinding, 2u> shape_bindings = bindings;
  shape_bindings[1].buffer = &wrong_shape;
  if (!EvidenceReason(
          rund::node::accel::RunAccelKernel(
              fixture.context, fixture.kernel,
              RunRequest(shape_bindings, fixture.host_input.size(), true)),
          "accel_kernel_buffer_shape_mismatch")) {
    return false;
  }

  const rund::AccelBuffer foreign_output = rund::node::accel::CreateAccelBuffer(
      other, BufferDesc(rund::BufferUsage::WriteOnly));
  if (!foreign_output.check.ok) {
    return false;
  }
  std::array<rund::AccelRunBinding, 2u> foreign_bindings = bindings;
  foreign_bindings[1].buffer = &foreign_output;
  if (!EvidenceReason(
          rund::node::accel::RunAccelKernel(
              fixture.context, fixture.kernel,
              RunRequest(foreign_bindings, fixture.host_input.size(), true)),
          "accel_kernel_buffer_owner_mismatch")) {
    return false;
  }

  return true;
}

bool LogicalAliasAdmissionMatches(const ResidentRunFixture &fixture) {
  const rund::compute_dsl::ComputeOp op = BuildFixedLane32Op();
  if (!op.ok()) {
    return false;
  }

  std::array<rund::AccelGraphBufferRef, 2u> collision_refs{
      rund::AccelRead(fixture.input, "input"),
      rund::AccelWrite(BufferDesc(rund::BufferUsage::WriteOnly), "output",
                       rund::GraphBufferVisibility::External, 1u),
  };
  std::array<rund::AccelGraphNode, 1u> collision_nodes{
      rund::AccelMap(op.ir(), collision_refs.data(), collision_refs.size(),
                     fixture.input.count)};
  const rund::AccelKernel collision_kernel =
      rund::node::accel::CompileAccelKernel(
          fixture.context, rund::AccelGraph{
                               .nodes = collision_nodes.data(),
                               .node_count = collision_nodes.size(),
                               .scalar = op.ir().scalar,
                               .domain = op.ir().domain,
                               .fixed_format = op.ir().fixed_format,
                           });
  if (!collision_kernel.check.ok) {
    return false;
  }
  const std::array<rund::AccelRunBinding, 2u> collision_bindings{
      rund::AccelRunBinding{
          .buffer = &fixture.input,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &fixture.output,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  if (!rund::node::accel::RunAccelKernel(
           fixture.context, collision_kernel,
           RunRequest(collision_bindings, fixture.input.count, true))
           .ok) {
    return false;
  }

  const rund::AccelBuffer intermediate = rund::node::accel::CreateAccelBuffer(
      fixture.context, BufferDesc(rund::BufferUsage::ReadWrite));
  const rund::AccelBuffer other_intermediate =
      rund::node::accel::CreateAccelBuffer(
          fixture.context, BufferDesc(rund::BufferUsage::ReadWrite));
  if (!intermediate.check.ok || !other_intermediate.check.ok) {
    return false;
  }
  std::array<rund::AccelGraphBufferRef, 4u> alias_refs{
      rund::AccelRead(BufferDesc(rund::BufferUsage::ReadOnly), "input",
                      rund::GraphBufferVisibility::External, 31u),
      rund::AccelWrite(BufferDesc(rund::BufferUsage::ReadWrite), "output",
                       rund::GraphBufferVisibility::External, 37u),
      rund::AccelRead(BufferDesc(rund::BufferUsage::ReadWrite), "input",
                      rund::GraphBufferVisibility::External, 37u),
      rund::AccelWrite(BufferDesc(rund::BufferUsage::WriteOnly), "output",
                       rund::GraphBufferVisibility::External, 41u),
  };
  std::array<rund::AccelGraphNode, 2u> alias_nodes{
      rund::AccelMap(op.ir(), alias_refs.data(), 2u, fixture.input.count),
      rund::AccelMap(op.ir(), alias_refs.data() + 2u, 2u, fixture.input.count),
  };
  const rund::AccelKernel alias_kernel = rund::node::accel::CompileAccelKernel(
      fixture.context, rund::AccelGraph{
                           .nodes = alias_nodes.data(),
                           .node_count = alias_nodes.size(),
                           .scalar = op.ir().scalar,
                           .domain = op.ir().domain,
                           .fixed_format = op.ir().fixed_format,
                       });
  if (!alias_kernel.check.ok) {
    return false;
  }
  std::array<rund::AccelRunBinding, 4u> alias_bindings{
      rund::AccelRunBinding{
          .buffer = &fixture.input,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &intermediate,
          .role = rund::kernel::BufferRole::Write,
      },
      rund::AccelRunBinding{
          .buffer = &intermediate,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &fixture.output,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  const auto request = [&](const auto &bindings) {
    return rund::AccelRun{
        .bindings = bindings.data(),
        .binding_count = bindings.size(),
        .tile_count = fixture.input.count,
        .fresh_evidence = true,
    };
  };
  if (!rund::node::accel::RunAccelKernel(fixture.context, alias_kernel,
                                         request(alias_bindings))
           .ok) {
    return false;
  }
  alias_bindings[2].buffer = &other_intermediate;
  return EvidenceReason(
      rund::node::accel::RunAccelKernel(fixture.context, alias_kernel,
                                        request(alias_bindings)),
      "accel_kernel_buffer_alias_mismatch");
}

bool IndexedWriteBoundaryMatches(const rund::AccelContext &context) {
  constexpr std::array<rund::kernel::i64, 6u> source{10, 20, 30, 40, 50, 60};
  constexpr std::array<rund::kernel::u32, 4u> invalid{5u, 6u, 3u, 0u};
  constexpr std::array<rund::kernel::u32, 4u> valid{5u, 1u, 3u, 0u};
  constexpr std::array<rund::kernel::i64, 4u> sentinel{-11, -22, -33, -44};
  constexpr std::array<rund::kernel::i64, 4u> expected{60, 20, 40, 10};
  const auto desc = [](const rund::BufferUsage usage, const std::uint64_t width,
                       const std::uint64_t count) {
    return rund::AccelBufferDesc{
        .scalar_width_bytes = width,
        .count = count,
        .usage = usage,
    };
  };
  const rund::AccelBuffer source_buffer = rund::node::accel::CreateAccelBuffer(
      context, desc(rund::BufferUsage::ReadOnly, 8u, source.size()));
  const rund::AccelBuffer index_buffer = rund::node::accel::CreateAccelBuffer(
      context, desc(rund::BufferUsage::ReadOnly, 4u, invalid.size()));
  const rund::AccelBuffer output_buffer = rund::node::accel::CreateAccelBuffer(
      context, desc(rund::BufferUsage::ReadWrite, 8u, sentinel.size()));
  if (!source_buffer.check.ok || !index_buffer.check.ok ||
      !output_buffer.check.ok ||
      !rund::node::accel::UploadAccelBuffer(context, source_buffer,
                                            source.data(), sizeof(source))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(context, index_buffer,
                                            invalid.data(), sizeof(invalid))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(context, output_buffer,
                                            sentinel.data(), sizeof(sentinel))
           .ok) {
    return false;
  }

  const rund::compute_dsl::ComputeOp op = BuildIndexedOp();
  if (!op.ok()) {
    return false;
  }
  std::array<rund::AccelGraphBufferRef, 3u> refs{
      rund::AccelRead(desc(rund::BufferUsage::ReadOnly, 8u, source.size()),
                      "source", rund::GraphBufferVisibility::External, 1u),
      rund::AccelRead(desc(rund::BufferUsage::ReadOnly, 4u, invalid.size()),
                      "index", rund::GraphBufferVisibility::External, 2u),
      rund::AccelWrite(desc(rund::BufferUsage::ReadWrite, 8u, sentinel.size()),
                       "output", rund::GraphBufferVisibility::External, 3u),
  };
  std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelMap(op.ir(), refs.data(), refs.size(), sentinel.size())};
  const rund::AccelKernel kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = op.ir().scalar,
                   .domain = op.ir().domain,
                   .fixed_format = op.ir().fixed_format,
               });
  if (!kernel.check.ok) {
    return false;
  }
  const std::array<rund::AccelRunBinding, 3u> bindings{
      rund::AccelRunBinding{.buffer = &source_buffer,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &index_buffer,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &output_buffer,
                            .role = rund::kernel::BufferRole::Write},
  };
  const auto run = [&] {
    return rund::node::accel::RunAccelKernel(
        context, kernel,
        rund::AccelRun{
            .bindings = bindings.data(),
            .binding_count = bindings.size(),
            .tile_count = sentinel.size(),
            .fresh_evidence = true,
        });
  };
  const rund::AccelEvidence rejected = run();
  std::array<rund::kernel::i64, sentinel.size()> observed{};
  if (rejected.ok ||
      std::string_view{rejected.reason} !=
          "compute_gather_index_out_of_range" ||
      !rund::node::accel::DownloadAccelBuffer(context, output_buffer,
                                              observed.data(), sizeof(observed))
           .ok ||
      observed != sentinel) {
    return false;
  }
  if (!rund::node::accel::UploadAccelBuffer(context, index_buffer, valid.data(),
                                            sizeof(valid))
           .ok ||
      !run().ok ||
      !rund::node::accel::DownloadAccelBuffer(context, output_buffer,
                                              observed.data(), sizeof(observed))
           .ok) {
    return false;
  }
  return observed == expected;
}

} // namespace node_accel_contract::kernel_case
