"""Link and run the SDK-disabled native Pipeline interface contract."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

try:
    from .model import CompileContext, CompileRow, flags_for_mode
    from .probe import ProbeError, compile_file, run, short_output
except ImportError:  # Direct invocation through contract.py.
    from model import CompileContext, CompileRow, flags_for_mode  # type: ignore[no-redef]
    from probe import ProbeError, compile_file, run, short_output  # type: ignore[no-redef]


_CONTRACT = r'''
#include <cstdio>
#include <string_view>

using namespace rund::node::accel::detail;

#define REQUIRE(condition)                                                \
  do {                                                                    \
    if (!(condition)) {                                                   \
      std::fprintf(stderr, "SDK-off Pipeline contract: %s\n", #condition); \
      return 1;                                                           \
    }                                                                     \
  } while (false)

bool unavailable(const rund::AccelCheck check, const char *reason) {
  return !check.ok && std::string_view{check.reason} == reason;
}

int main() {
  constexpr auto metal = "accel_metal_unavailable";
  bool ready = true;
  REQUIRE(unavailable(MetalPipelineResidencyReady({}, ready), metal));
  REQUIRE(!ready);
  for (const auto memory : {ResidencySlidingMemory::HostCoherent,
                           ResidencySlidingMemory::NoncoherentIntegratedCopy,
                           ResidencySlidingMemory::NoncoherentSplitTransfer}) {
    const auto capability = MetalResidencySlidingCapability({}, memory);
    REQUIRE(unavailable(capability.check, metal));
    REQUIRE(capability.memory == memory && capability.max_slots == 0u);
    REQUIRE(capability.retained_bytes == 0u && !capability.callbacks_async);
  }
  unsigned callbacks = 0u;
  const KernelCompletion completion = [](void *user, KernelResult) noexcept {
    ++*static_cast<unsigned *>(user);
  };
  REQUIRE(unavailable(SubmitMetalResidencySliding(
                          {}, {}, completion, &callbacks, KernelTiming::None,
                          PipelineSubmitMode::Standard, {}), metal));
  REQUIRE(callbacks == 0u);
  REQUIRE(unavailable(AbortMetalResidencyWindow({}, {}), metal));
  REQUIRE(!InjectMetalResidencySlidingStaleDescriptorOnce({}));

  MetalResidencySlidingDiagnostics sliding{};
  sliding.retained_bytes = 0x100000001ull;
  sliding.ready = true;
  REQUIRE(!InspectMetalResidencySliding({}, sliding));
  REQUIRE(sliding.retained_bytes == 0u && !sliding.ready);

  const auto prepared = PrepareMetalPersistentResidencySliding({});
  REQUIRE(unavailable(prepared.capability.check, metal));
  REQUIRE(!prepared.lowering && !prepared.cell && !prepared.ticket);
  REQUIRE(prepared.encoded_coordinate_count == 0u);
  const auto &service = MetalPersistentResidencySlidingServiceOps();
  REQUIRE(!service.submit_result && !service.wait_done && !service.signal_ready);
  REQUIRE(!service.ack_done && !service.fail_service && !service.quarantine_unknown);

  MetalPersistentResidencySlidingDiagnostics persistent{};
  persistent.encoded_coordinate_count = 0x100000001ull;
  persistent.submitted = true;
  REQUIRE(!InspectMetalPersistentResidencySliding({}, persistent));
  REQUIRE(persistent.encoded_coordinate_count == 0u && !persistent.submitted);

  MetalFusedDirectRecurrenceDiagnostics metal_recurrence{};
  metal_recurrence.iteration_count = 0x100000001ull;
  metal_recurrence.retention = MetalFusedDirectRecurrenceRetention::History;
  REQUIRE(!InspectMetalFusedDirectRecurrence({}, metal_recurrence));
  REQUIRE(metal_recurrence.iteration_count == 0u);
  REQUIRE(metal_recurrence.retention == MetalFusedDirectRecurrenceRetention::Unknown);

  VulkanFusedDirectRecurrenceDiagnostics vulkan_recurrence{};
  vulkan_recurrence.iteration_count = 0x100000001ull;
  vulkan_recurrence.retention = VulkanFusedDirectRecurrenceRetention::History;
  REQUIRE(!InspectVulkanFusedDirectRecurrence({}, vulkan_recurrence));
  REQUIRE(vulkan_recurrence.iteration_count == 0u);
  REQUIRE(vulkan_recurrence.retention == VulkanFusedDirectRecurrenceRetention::Unknown);
  return 0;
}
'''


def run_availability_probe(
    context: CompileContext,
    rows: Sequence[CompileRow],
    root: Path,
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    by_source = {row.source: row for row in rows}
    objects: list[Path] = []
    includes: list[str] = []
    for backend in ("metal", "vulkan"):
        owner = root / "node" / "src" / "accel" / backend
        source = owner / "kernel" / "pipeline" / "stub.cpp"
        row = by_source.get(source)
        if row is None:
            raise ProbeError(f"SDK-off Pipeline owner absent from compile database: {source}")
        native = CompileContext(backend, row.source, row.directory, row.compiler, row.flags)
        output = directory / (backend + ".o")
        result = compile_file(native, flags_for_mode(native, "off"), source, output)
        if result.returncode != 0:
            raise ProbeError(f"SDK-off {backend} Pipeline compile failed: {short_output(result.output)}")
        objects.append(output)
        includes.append(f'#include "{(owner / "kernel.hpp").as_posix()}"\n')

    source = directory / "availability.cpp"
    source.write_text("".join(includes) + _CONTRACT, encoding="utf-8")
    executable = directory / "availability"
    result = run(
        (context.compiler, *flags_for_mode(context, "off"), "-O0", "-fno-lto",
         str(source), *(str(path) for path in objects), "-o", str(executable)),
        cwd=context.directory,
    )
    if result.returncode != 0:
        raise ProbeError(f"SDK-off Pipeline executable link failed: {short_output(result.output)}")
    result = run((str(executable),), cwd=directory, timeout=30)
    if result.returncode != 0:
        raise ProbeError(f"SDK-off Pipeline execution failed: {short_output(result.output)}")
