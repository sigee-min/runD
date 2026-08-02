#pragma once

#include "../callback.hpp"
#include "../memory.hpp"
#include "../prepared/failure.hpp"
#include "run.hpp"

#include <accel/check.hpp>

#include <memory>
#include <span>

namespace rund::node::accel::detail {

struct TileTransducer;
struct NestedAggregate;
struct PreparedKernelRouteReservation;
struct PreparedKernelPipelineReservation;
struct PreparedKernelTemplateRegistry;
struct PreparedMapRecurrenceReservation;
struct MapRecurrencePreparationPlan;

[[nodiscard]] rund::AccelCheck RunFakeKernel(const BackendRun &run);
[[nodiscard]] rund::AccelCheck RunMetalKernel(const BackendRun &run);
[[nodiscard]] rund::AccelCheck RunVulkanKernel(const BackendRun &run);

[[nodiscard]] rund::AccelCheck
PrepareFakeKernel(const BackendRun &run, std::shared_ptr<void> &prepared,
                  PreparedMemory &memory);
[[nodiscard]] rund::AccelCheck SubmitPreparedFakeKernel(
    const BackendRun &run, const std::shared_ptr<void> &prepared,
    KernelCompletion completion, void *user, PreparedMemoryMeter *memory,
    const std::shared_ptr<void> &lifetime) noexcept;

[[nodiscard]] rund::AccelCheck
PrepareMetalKernel(const BackendRun &run, std::shared_ptr<void> &prepared,
                   PreparedMemory &memory);
[[nodiscard]] rund::AccelCheck
PrepareMetalPipelinePrivateKernel(const BackendRun &run,
                                  std::shared_ptr<void> &prepared,
                                  PreparedMemory &memory);
[[nodiscard]] rund::AccelCheck PlanMetalPipelinePrivateKernel(
    const BackendRun &run,
    PreparedKernelRouteReservation &reservation) noexcept;
[[nodiscard]] rund::AccelCheck PlanMetalPipelineProgram(
    const KernelExecution &execution, const PreparedKernelProgramRoute &route,
    PreparedKernelRouteReservation &reservation) noexcept;
[[nodiscard]] rund::AccelCheck PlanMetalPipelineRecurrence(
    const MapRecurrencePreparationPlan &plan,
    PreparedMapRecurrenceReservation &reservation) noexcept;
[[nodiscard]] rund::AccelCheck PlanMetalPipelineStructure(
    const rund::AccelContext &context,
    PreparedKernelPipelineReservation &reservation) noexcept;
[[nodiscard]] bool SameMetalPipelineProgramTemplate(
    const KernelExecution &execution, const PreparedKernelProgramRoute &left,
    const PreparedKernelProgramRoute &right) noexcept;
[[nodiscard]] bool SameMetalPipelineTemplate(const BackendRun &left,
                                             const BackendRun &right) noexcept;
[[nodiscard]] rund::AccelCheck
RunPreparedMetalBatch(std::span<const BackendBatchEntry> entries,
                      std::span<rund::AccelCheck> results,
                      std::shared_ptr<void> &workspace,
                      rund::RuntimeStats &stats);
[[nodiscard]] rund::AccelCheck
PrepareMetalPipeline(std::span<const BackendBatchEntry> templates,
                     std::span<const BackendBatchEntry> entries,
                     std::span<const std::uint8_t> barriers,
                     std::span<const TileTransducer> transducers,
                     std::span<const NestedAggregate> aggregates,
                     std::span<const BackendPublish> publications,
                     PreparedKernelTemplateRegistry &registry,
                     PreparedPipelineStatusLayout &status,
                     bool profile_steps, std::shared_ptr<void> &prepared,
                     PreparedPipelineMemory &memory,
                     PreparedPipelineFailure &failure);
[[nodiscard]] rund::AccelCheck
SubmitPreparedMetalPipeline(const std::shared_ptr<void> &prepared,
                            KernelCompletion completion, void *user) noexcept;
[[nodiscard]] rund::AccelCheck SubmitPreparedMetalKernel(
    const BackendRun &run, const std::shared_ptr<void> &prepared,
    KernelCompletion completion, void *user, PreparedMemoryMeter *memory,
    const std::shared_ptr<void> &lifetime) noexcept;

[[nodiscard]] rund::AccelCheck
PrepareVulkanKernel(const BackendRun &run, std::shared_ptr<void> &prepared,
                    PreparedMemory &memory);
[[nodiscard]] rund::AccelCheck
PrepareVulkanPipelinePrivateKernel(const BackendRun &run,
                                   std::shared_ptr<void> &prepared,
                                   PreparedMemory &memory);
[[nodiscard]] rund::AccelCheck PlanVulkanPipelinePrivateKernel(
    const BackendRun &run,
    PreparedKernelRouteReservation &reservation) noexcept;
[[nodiscard]] rund::AccelCheck PlanVulkanPipelineProgram(
    const KernelExecution &execution, const PreparedKernelProgramRoute &route,
    PreparedKernelRouteReservation &reservation) noexcept;
[[nodiscard]] rund::AccelCheck PlanVulkanPipelineRecurrence(
    const MapRecurrencePreparationPlan &plan,
    PreparedMapRecurrenceReservation &reservation) noexcept;
[[nodiscard]] rund::AccelCheck PlanVulkanPipelineStructure(
    const rund::AccelContext &context,
    PreparedKernelPipelineReservation &reservation) noexcept;
[[nodiscard]] bool SameVulkanPipelineProgramTemplate(
    const KernelExecution &execution, const PreparedKernelProgramRoute &left,
    const PreparedKernelProgramRoute &right) noexcept;
[[nodiscard]] bool
SameVulkanPipelineTemplate(const BackendRun &left,
                           const BackendRun &right) noexcept;
[[nodiscard]] rund::AccelCheck
RunPreparedVulkanBatch(std::span<const BackendBatchEntry> entries,
                       std::span<rund::AccelCheck> results,
                       std::shared_ptr<void> &workspace,
                       rund::RuntimeStats &stats);
[[nodiscard]] rund::AccelCheck
PrepareVulkanPipeline(std::span<const BackendBatchEntry> templates,
                      std::span<const BackendBatchEntry> entries,
                      std::span<const std::uint8_t> barriers,
                      std::span<const TileTransducer> transducers,
                      std::span<const NestedAggregate> aggregates,
                      std::span<const BackendPublish> publications,
                      PreparedKernelTemplateRegistry &registry,
                      PreparedPipelineStatusLayout &status,
                      bool profile_steps, std::shared_ptr<void> &prepared,
                      PreparedPipelineMemory &memory,
                      PreparedPipelineFailure &failure);
[[nodiscard]] rund::AccelCheck
SubmitPreparedVulkanPipeline(const std::shared_ptr<void> &prepared,
                             KernelCompletion completion, void *user) noexcept;
[[nodiscard]] rund::AccelCheck SubmitPreparedVulkanKernel(
    const BackendRun &run, const std::shared_ptr<void> &prepared,
    KernelCompletion completion, void *user, PreparedMemoryMeter *memory,
    const std::shared_ptr<void> &lifetime) noexcept;

} // namespace rund::node::accel::detail
