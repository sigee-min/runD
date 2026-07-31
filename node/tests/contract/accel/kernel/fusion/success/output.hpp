#pragma once

#include "../local.hpp"

#include <accel/graph/factory/map.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/value.hpp>

#include <node/accel/context.hpp>

#include "src/accel/context/internal.hpp"

#include <array>
#include <cstddef>
#include <cstdio>
#include <string_view>

namespace node_accel_contract::fusion {

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildWideProducer() {
  std::array<rund::kernel::i64, 8u> input{};
  std::array<rund::kernel::i64, 8u> output{};
  const auto body = rund::compute_dsl::bind(input.size())
                        .fixed<20, 44>()
                        .param<"dt">(rund::kernel::i64{7})
                        .read<"input">(input.data())
                        .write<"output">(output.data());
  return rund::compute_dsl::def("fusion-output-producer-wide")
      .on(body)
      .map([](auto i, auto b) {
        const auto dt = b.template param<"dt">();
        const auto input = b.template read<"input">();
        auto output = b.template write<"output">();
        output[i] = input[i] + dt;
      });
}

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildWideTerminal() {
  std::array<rund::kernel::i64, 8u> input{};
  std::array<rund::kernel::i64, 8u> first{};
  std::array<rund::kernel::i64, 8u> second{};
  std::array<rund::kernel::i64, 8u> third{};
  const auto body = rund::compute_dsl::bind(input.size())
                        .fixed<20, 44>()
                        .read<"input">(input.data())
                        .write<"first">(first.data())
                        .write<"second">(second.data())
                        .write<"third">(third.data());
  return rund::compute_dsl::def("fusion-output-terminal-wide")
      .on(body)
      .map([](auto i, auto b) {
        const auto input = b.template read<"input">();
        b.template write<"first">()[i] = input[i] + 1;
        b.template write<"second">()[i] = input[i] + 2;
        b.template write<"third">()[i] = input[i] + 3;
      });
}

template <typename Raw>
[[nodiscard]] inline rund::AccelBuffer
MakeOutputBuffer(const rund::AccelContext &context,
                 const rund::BufferUsage usage) {
  return rund::node::accel::CreateAccelBuffer(
      context, rund::AccelBufferDesc{.scalar_width_bytes = sizeof(Raw),
                                     .count = 8u,
                                     .usage = usage});
}

template <typename Raw>
[[nodiscard]] bool
RunTerminalOutputs(const rund::AccelContext &context,
                   const rund::compute_dsl::ComputeOp &producer,
                   const rund::compute_dsl::ComputeOp &terminal) {
  constexpr std::size_t count = 8u;
  const std::array<Raw, count> source{Raw{-20}, Raw{-3}, Raw{0},  Raw{2},
                                      Raw{9},   Raw{25}, Raw{64}, Raw{100}};
  std::array<Raw, count> expected_first{};
  std::array<Raw, count> expected_second{};
  std::array<Raw, count> expected_third{};
  for (std::size_t index = 0u; index < count; ++index) {
    expected_first[index] = static_cast<Raw>(source[index] + Raw{8});
    expected_second[index] = static_cast<Raw>(source[index] + Raw{9});
    expected_third[index] = static_cast<Raw>(source[index] + Raw{10});
  }

  if (!producer.ok() || !terminal.ok()) {
    return false;
  }

  const rund::AccelBuffer fused_input =
      MakeOutputBuffer<Raw>(context, rund::BufferUsage::ReadOnly);
  const rund::AccelBuffer fused_middle =
      MakeOutputBuffer<Raw>(context, rund::BufferUsage::ReadWrite);
  const rund::AccelBuffer fused_first =
      MakeOutputBuffer<Raw>(context, rund::BufferUsage::WriteOnly);
  const rund::AccelBuffer fused_second =
      MakeOutputBuffer<Raw>(context, rund::BufferUsage::WriteOnly);
  const rund::AccelBuffer fused_third =
      MakeOutputBuffer<Raw>(context, rund::BufferUsage::WriteOnly);
  const rund::AccelBuffer oracle_input =
      MakeOutputBuffer<Raw>(context, rund::BufferUsage::ReadOnly);
  const rund::AccelBuffer oracle_middle =
      MakeOutputBuffer<Raw>(context, rund::BufferUsage::ReadWrite);
  const rund::AccelBuffer oracle_first =
      MakeOutputBuffer<Raw>(context, rund::BufferUsage::WriteOnly);
  const rund::AccelBuffer oracle_second =
      MakeOutputBuffer<Raw>(context, rund::BufferUsage::WriteOnly);
  const rund::AccelBuffer oracle_third =
      MakeOutputBuffer<Raw>(context, rund::BufferUsage::WriteOnly);
  const std::array<const rund::AccelBuffer *, 10u> buffers{
      &fused_input,   &fused_middle, &fused_first,   &fused_second,
      &fused_third,   &oracle_input, &oracle_middle, &oracle_first,
      &oracle_second, &oracle_third};
  for (const rund::AccelBuffer *const buffer : buffers) {
    if (!buffer->check.ok) {
      std::fprintf(stderr, "terminal output buffer failed api=%u width=%zu\n",
                   static_cast<unsigned>(context.api), sizeof(Raw) * 8u);
      return false;
    }
  }
  if (!rund::node::accel::UploadAccelBuffer(context, fused_input, source.data(),
                                            sizeof(source))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(context, oracle_input,
                                            source.data(), sizeof(source))
           .ok) {
    return false;
  }

  std::array<GraphBufferRef, 6u> fused_refs{
      GraphBufferRef{.buffer = &fused_input, .role = Role::Read},
      GraphBufferRef{.buffer = &fused_middle,
                     .role = Role::Write,
                     .visibility = Visibility::Internal},
      GraphBufferRef{.buffer = &fused_middle,
                     .role = Role::Read,
                     .binding_name = "input",
                     .visibility = Visibility::Internal},
      GraphBufferRef{
          .buffer = &fused_first, .role = Role::Write, .binding_name = "first"},
      GraphBufferRef{.buffer = &fused_second,
                     .role = Role::Write,
                     .binding_name = "second"},
      GraphBufferRef{
          .buffer = &fused_third, .role = Role::Write, .binding_name = "third"},
  };
  std::array<GraphNode, 2u> fused_nodes{
      rund::AccelMap(producer.ir(), fused_refs.data(), 2u, count),
      rund::AccelMap(terminal.ir(), fused_refs.data() + 2u, 4u, count),
  };
  const rund::AccelKernel fused = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = fused_nodes.data(),
                   .node_count = fused_nodes.size(),
                   .scalar = producer.ir().scalar,
                   .domain = producer.ir().domain,
                   .fixed_format = producer.ir().fixed_format,
               });
  if (!fused.check.ok) {
    std::fprintf(stderr,
                 "terminal output fusion compile failed api=%u width=%zu "
                 "reason=%s\n",
                 static_cast<unsigned>(context.api), sizeof(Raw) * 8u,
                 fused.check.reason);
    return false;
  }

  const rund::node::accel::detail::KernelExecution execution =
      rund::node::accel::detail::AdmitKernelForExecution(context, fused);
  if (!execution.admission.check.ok || execution.steps.size() != 1u ||
      execution.original_operation_count != 2u ||
      execution.fused_operation_count != 1u ||
      execution.removed_dispatch_count != 1u ||
      execution.fusion_rejection_count != 0u) {
    std::fprintf(
        stderr,
        "terminal output admission failed api=%u width=%zu reason=%s "
        "steps=%zu original=%llu fused=%llu removed=%llu rejects=%llu\n",
        static_cast<unsigned>(context.api), sizeof(Raw) * 8u,
        execution.admission.check.reason, execution.steps.size(),
        static_cast<unsigned long long>(execution.original_operation_count),
        static_cast<unsigned long long>(execution.fused_operation_count),
        static_cast<unsigned long long>(execution.removed_dispatch_count),
        static_cast<unsigned long long>(execution.fusion_rejection_count));
    return false;
  }
  const auto &step = execution.steps.front();
  const auto &indices = step.graph_binding_indices;
  if (!step.artifact.ok || step.artifact.metadata.read_count != 1u ||
      step.artifact.metadata.write_count != 3u ||
      !step.graph_binding_indices_ok || !indices.valid() ||
      indices.size() != 4u || indices[0] != 0u || indices[1] != 3u ||
      indices[2] != 4u || indices[3] != 5u) {
    std::fprintf(
        stderr,
        "terminal output projection failed api=%u width=%zu "
        "artifact=%d reads=%llu writes=%llu indices=%llu\n",
        static_cast<unsigned>(context.api), sizeof(Raw) * 8u,
        step.artifact.ok ? 1 : 0,
        static_cast<unsigned long long>(step.artifact.metadata.read_count),
        static_cast<unsigned long long>(step.artifact.metadata.write_count),
        static_cast<unsigned long long>(indices.size()));
    return false;
  }

  std::array<KernelBinding, 6u> fused_bindings{
      KernelBinding{.buffer = &fused_input, .role = Role::Read},
      KernelBinding{.buffer = &fused_middle, .role = Role::Write},
      KernelBinding{.buffer = &fused_middle, .role = Role::Read},
      KernelBinding{.buffer = &fused_first, .role = Role::Write},
      KernelBinding{.buffer = &fused_second, .role = Role::Write},
      KernelBinding{.buffer = &fused_third, .role = Role::Write},
  };
  const rund::AccelEvidence evidence = rund::node::accel::RunAccelKernel(
      context, fused,
      rund::AccelRun{.bindings = fused_bindings.data(),
                     .binding_count = fused_bindings.size(),
                     .tile_count = count,
                     .fresh_evidence = true});
  if (!evidence.ok || evidence.original_operation_count != 2u ||
      evidence.fused_operation_count != 1u ||
      evidence.original_dispatch_count != 2u ||
      evidence.final_dispatch_count != 1u || evidence.dispatch_count != 1u ||
      evidence.fusion_rejection_count != 0u ||
      std::string_view{evidence.fusion_reason} != "compute_fusion_ok") {
    std::fprintf(
        stderr,
        "terminal output fused run failed api=%u width=%zu "
        "reason=%s original=%llu fused=%llu original_dispatch=%llu "
        "final_dispatch=%llu dispatch=%llu rejects=%llu\n",
        static_cast<unsigned>(context.api), sizeof(Raw) * 8u, evidence.reason,
        static_cast<unsigned long long>(evidence.original_operation_count),
        static_cast<unsigned long long>(evidence.fused_operation_count),
        static_cast<unsigned long long>(evidence.original_dispatch_count),
        static_cast<unsigned long long>(evidence.final_dispatch_count),
        static_cast<unsigned long long>(evidence.dispatch_count),
        static_cast<unsigned long long>(evidence.fusion_rejection_count));
    return false;
  }

  std::array<GraphBufferRef, 2u> producer_refs{
      GraphBufferRef{.buffer = &oracle_input, .role = Role::Read},
      GraphBufferRef{.buffer = &oracle_middle, .role = Role::Write},
  };
  std::array<GraphNode, 1u> producer_nodes{rund::AccelMap(
      producer.ir(), producer_refs.data(), producer_refs.size(), count)};
  std::array<GraphBufferRef, 4u> terminal_refs{
      GraphBufferRef{.buffer = &oracle_middle,
                     .role = Role::Read,
                     .binding_name = "input"},
      GraphBufferRef{.buffer = &oracle_first,
                     .role = Role::Write,
                     .binding_name = "first"},
      GraphBufferRef{.buffer = &oracle_second,
                     .role = Role::Write,
                     .binding_name = "second"},
      GraphBufferRef{.buffer = &oracle_third,
                     .role = Role::Write,
                     .binding_name = "third"},
  };
  std::array<GraphNode, 1u> terminal_nodes{rund::AccelMap(
      terminal.ir(), terminal_refs.data(), terminal_refs.size(), count)};
  const rund::AccelKernel oracle_producer =
      rund::node::accel::CompileAccelKernel(
          context,
          rund::AccelGraph{.nodes = producer_nodes.data(),
                           .node_count = producer_nodes.size(),
                           .scalar = producer.ir().scalar,
                           .domain = producer.ir().domain,
                           .fixed_format = producer.ir().fixed_format});
  const rund::AccelKernel oracle_terminal =
      rund::node::accel::CompileAccelKernel(
          context,
          rund::AccelGraph{.nodes = terminal_nodes.data(),
                           .node_count = terminal_nodes.size(),
                           .scalar = terminal.ir().scalar,
                           .domain = terminal.ir().domain,
                           .fixed_format = terminal.ir().fixed_format});
  if (!oracle_producer.check.ok || !oracle_terminal.check.ok) {
    return false;
  }

  std::array<KernelBinding, 2u> producer_bindings{
      KernelBinding{.buffer = &oracle_input, .role = Role::Read},
      KernelBinding{.buffer = &oracle_middle, .role = Role::Write},
  };
  std::array<KernelBinding, 4u> terminal_bindings{
      KernelBinding{.buffer = &oracle_middle, .role = Role::Read},
      KernelBinding{.buffer = &oracle_first, .role = Role::Write},
      KernelBinding{.buffer = &oracle_second, .role = Role::Write},
      KernelBinding{.buffer = &oracle_third, .role = Role::Write},
  };
  const rund::AccelEvidence producer_run = rund::node::accel::RunAccelKernel(
      context, oracle_producer,
      rund::AccelRun{.bindings = producer_bindings.data(),
                     .binding_count = producer_bindings.size(),
                     .tile_count = count,
                     .fresh_evidence = true});
  const rund::AccelEvidence terminal_run = rund::node::accel::RunAccelKernel(
      context, oracle_terminal,
      rund::AccelRun{.bindings = terminal_bindings.data(),
                     .binding_count = terminal_bindings.size(),
                     .tile_count = count,
                     .fresh_evidence = true});
  if (!producer_run.ok || !terminal_run.ok) {
    return false;
  }

  std::array<Raw, count> fused_first_out{};
  std::array<Raw, count> fused_second_out{};
  std::array<Raw, count> fused_third_out{};
  std::array<Raw, count> oracle_first_out{};
  std::array<Raw, count> oracle_second_out{};
  std::array<Raw, count> oracle_third_out{};
  const auto download = [&](const rund::AccelBuffer &buffer, auto &output) {
    return rund::node::accel::DownloadAccelBuffer(context, buffer,
                                                  output.data(), sizeof(output))
        .ok;
  };
  if (!download(fused_first, fused_first_out) ||
      !download(fused_second, fused_second_out) ||
      !download(fused_third, fused_third_out) ||
      !download(oracle_first, oracle_first_out) ||
      !download(oracle_second, oracle_second_out) ||
      !download(oracle_third, oracle_third_out)) {
    return false;
  }
  for (std::size_t index = 0u; index < count; ++index) {
    const bool equal = fused_first_out[index] == oracle_first_out[index] &&
                       fused_second_out[index] == oracle_second_out[index] &&
                       fused_third_out[index] == oracle_third_out[index] &&
                       fused_first_out[index] == expected_first[index] &&
                       fused_second_out[index] == expected_second[index] &&
                       fused_third_out[index] == expected_third[index];
    if (!equal) {
      std::fprintf(stderr,
                   "terminal output parity failed api=%u width=%zu index=%zu "
                   "fused=%lld,%lld,%lld oracle=%lld,%lld,%lld "
                   "expected=%lld,%lld,%lld\n",
                   static_cast<unsigned>(context.api), sizeof(Raw) * 8u, index,
                   static_cast<long long>(fused_first_out[index]),
                   static_cast<long long>(fused_second_out[index]),
                   static_cast<long long>(fused_third_out[index]),
                   static_cast<long long>(oracle_first_out[index]),
                   static_cast<long long>(oracle_second_out[index]),
                   static_cast<long long>(oracle_third_out[index]),
                   static_cast<long long>(expected_first[index]),
                   static_cast<long long>(expected_second[index]),
                   static_cast<long long>(expected_third[index]));
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool
TerminalOutputsPreserved(const rund::AccelContext &context) {
  return RunTerminalOutputs<rund::kernel::i32>(context, BuildFixedLane32Op(),
                                               BuildTerminalWriteOp()) &&
         RunTerminalOutputs<rund::kernel::i64>(context, BuildWideProducer(),
                                               BuildWideTerminal());
}

} // namespace node_accel_contract::fusion
