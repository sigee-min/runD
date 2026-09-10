#include "../backing.hpp"

#include "../cache.hpp"

#include "../../../device/residency/pool.hpp"
#include "../../../device/state.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/pipeline/shape.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <utility>

namespace rund::compute::detail {

Status schedule_virtual_prefetch(
    VirtualBacking &backing, const VirtualEpochProjection &epoch,
    const VirtualRunProjection &run, residency::Pool &pool,
    residency::Prefetcher &prefetcher, const bool speculative, bool &pending,
    bool &cleanup_failed, const std::span<std::byte> coherent_input,
    const bool coherent_deferred) noexcept {
  pending = false;
  if (pool.device == nullptr || pool.device->backend == Backend::Cpu) {
    return pool.device == nullptr ? Status::fail(Reason::DeviceInvalid)
                                  : Status::success();
  }
  if (epoch.page_count == 0u || epoch.page_count > PipelineLeafCapacity) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::array<residency::CacheUse, PipelineLeafCapacity> uses{};
  std::array<residency::CacheUse, PipelineLeafCapacity> outputs{};
  std::array<residency::CacheKey, PipelineLeafCapacity> keys{};
  std::array<std::uint8_t, PipelineLeafCapacity> device_resident{};
  std::array<residency::PrefetchRequest, PipelineLeafCapacity> requests{};
  std::array<residency::AliasLease, PipelineLeafCapacity> aliases{};
  std::size_t alias_count = 0u;
  std::array<VirtualInputPageProjection, PipelineLeafCapacity>
      projected_pages{};
  const std::size_t page_count = static_cast<std::size_t>(epoch.page_count);
  for (std::size_t index = 0u; index < page_count; ++index) {
    const std::uint64_t page = epoch.failed_page + index;
    std::uint64_t next_use = residency::NeverUse;
    if (!run.active.stream.next_use(page, next_use)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    keys[index] =
        virtual_cache_key(run, run.input_backing, run.input_version, page);
    uses[index] = residency::CacheUse{.key = keys[index],
                                      .access = residency::Access::Read,
                                      .next_use = next_use};
  }
  if (run.frame_capacity == 0u ||
      epoch.failed_page % run.frame_capacity != 0u ||
      run.host_frame_capacity < run.frame_capacity ||
      pool.host_input_frame_count !=
          run.host_frame_capacity * residency::Pool::BankCount) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::uint32_t bank = static_cast<std::uint32_t>(
      (epoch.failed_page / run.frame_capacity) % residency::Pool::BankCount);
  const residency::FrameRegion device_region = pool.input_regions[bank];
  if (device_region.count != run.frame_capacity) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const bool coherent = !coherent_input.empty();
  std::uint64_t input_bytes = 0u;
  if (coherent &&
      (run.scan() || run.reduction() || run.graph_reduction() ||
       !kernel::checked::mul(run.input_page_bytes, device_region.count,
                             input_bytes) ||
       input_bytes != coherent_input.size() ||
       !project_residency_transform_uses(
           epoch, run, std::span<residency::CacheUse>{uses.data(), page_count},
           std::span<residency::CacheUse>{outputs.data(), page_count}))) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (!coherent && !run.scan() &&
      !pool.authority().probe(
          std::span<const residency::CacheKey>{keys.data(), page_count},
          std::span<std::uint8_t>{device_resident.data(), page_count},
          device_region.first, device_region.count)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::size_t request_count = 0u;
  for (std::size_t index = 0u; index < page_count; ++index) {
    if (coherent || device_resident[index] == 0u) {
      uses[request_count++] = uses[index];
    }
  }
  if (request_count == 0u) {
    return Status::success();
  }
  const std::uint32_t host_bank_first =
      pool.first_host_input_frame + bank * run.host_frame_capacity;
  const residency::AuthorityResult acquired =
      coherent
          ? pool.authority().begin_transform(
                std::span<const residency::CacheUse>{uses.data(),
                                                     request_count},
                device_region,
                std::span<const residency::CacheUse>{outputs.data(),
                                                     request_count},
                residency::FrameRegion{
                    .tier = residency::FrameTier::Device,
                    .role = residency::FrameRole::Output,
                    .first =
                        pool.first_output_frame +
                        bank * static_cast<std::uint32_t>(run.frame_capacity),
                    .count = static_cast<std::uint32_t>(run.frame_capacity),
                })
          : pool.authority().begin(
                std::span<const residency::CacheUse>{uses.data(),
                                                     request_count},
                residency::FrameTier::Host, residency::FrameRole::Input,
                host_bank_first, run.host_frame_capacity);
  if (!acquired) {
    return Status::fail(acquired.failure == residency::AuthorityFailure::Busy
                            ? Reason::PipelineBusy
                            : Reason::PipelineMemoryBudget);
  }
  const auto abort = [&]() noexcept {
    const bool cleaned = pool.authority().complete(acquired.lease.token, false);
    cleanup_failed = !cleaned || cleanup_failed;
    return Status::fail(cleaned ? Reason::PipelineInvalid
                                : Reason::PipelineBusy);
  };
  if (acquired.lease.bindings.size() !=
          request_count * static_cast<std::size_t>(coherent ? 2u : 1u) ||
      std::any_of(
          acquired.lease.transitions.begin(), acquired.lease.transitions.end(),
          [](const residency::CacheTransition transition) {
            return transition.kind == residency::TransitionKind::Writeback;
          })) {
    return abort();
  }
  for (std::size_t index = 0u; index < request_count; ++index) {
    const residency::CacheBinding binding = acquired.lease.bindings[index];
    VirtualInputPageProjection &projected = projected_pages[index];
    const std::uint32_t expected_first =
        coherent ? device_region.first : host_bank_first;
    const std::uint32_t expected_count =
        coherent ? device_region.count : run.host_frame_capacity;
    if (binding.frame < expected_first ||
        binding.frame >= expected_first + expected_count ||
        !project_virtual_input_page(run, binding.key.page, projected)) {
      return abort();
    }
    const std::size_t local = binding.frame - expected_first;
    std::byte *const target =
        coherent ? coherent_input.data() + local * run.input_page_bytes
                 : virtual_host_input_frame(run, binding.frame);
    if (target == nullptr) {
      return abort();
    }
    requests[index] = residency::PrefetchRequest{
        .key = binding.key,
        .offset = projected.logical_offset,
        .bytes = projected.transfer_bytes,
        .target_offset = projected.target_offset,
        .read_offset = projected.logical_offset,
        .read_bytes = projected.transfer_bytes,
        .read_target_offset = projected.target_offset,
        .frame = target,
        .physical_frame = binding.frame,
        .fetch = binding.fetch,
    };
    VirtualInputReuseProjection reuse{};
    if (index != 0u && binding.fetch &&
        requests[index - 1u].key.page !=
            std::numeric_limits<std::uint64_t>::max() &&
        requests[index - 1u].key.page + 1u == binding.key.page &&
        requests[index - 1u].frame != target &&
        project_virtual_input_reuse(projected_pages[index - 1u], projected,
                                    reuse)) {
      requests[index].read_offset = reuse.logical_offset;
      requests[index].read_bytes = reuse.read_bytes;
      requests[index].read_target_offset = reuse.read_target_offset;
      requests[index].reuse_source_offset = reuse.source_offset;
      requests[index].reuse_target_offset = reuse.target_offset;
      requests[index].reuse_bytes = reuse.bytes;
    }
  }
  std::uint32_t target_bank = 0u;
  const bool valid_target_bank =
      run.frame_capacity != 0u && epoch.failed_page / run.frame_capacity <=
                                      std::numeric_limits<std::uint32_t>::max();
  if (valid_target_bank) {
    target_bank = static_cast<std::uint32_t>(
        (epoch.failed_page / run.frame_capacity) % residency::Pool::BankCount);
  }
  VirtualInputPageProjection expected_first_page{};
  const bool exact_first_page =
      request_count != 0u && requests[0u].fetch &&
      requests[0u].key.page == epoch.failed_page &&
      project_virtual_input_page(run, epoch.failed_page, expected_first_page) &&
      projected_pages[0u].logical_offset ==
          expected_first_page.logical_offset &&
      projected_pages[0u].target_offset == expected_first_page.target_offset &&
      projected_pages[0u].transfer_bytes ==
          expected_first_page.transfer_bytes &&
      projected_pages[0u].leading_fill_bytes ==
          expected_first_page.leading_fill_bytes &&
      projected_pages[0u].trailing_fill_offset ==
          expected_first_page.trailing_fill_offset;
  if (!coherent && !run.graph_execution() && !run.reduction() && !run.scan() &&
      epoch.failed_page != 0u && valid_target_bank && exact_first_page) {
    VirtualInputPageProjection prior{};
    if (project_virtual_input_page(run, epoch.failed_page - 1u, prior)) {
      VirtualInputReuseProjection reuse{};
      if (project_virtual_input_reuse(prior, projected_pages[0u], reuse)) {
        std::array<residency::FrameRegion, residency::Pool::BankCount>
            source_regions{};
        bool valid_sources = true;
        for (std::size_t bank = 0u; bank < residency::Pool::BankCount; ++bank) {
          source_regions[bank] = virtual_graph_host_input_region(
              run, 0u, static_cast<std::uint32_t>(bank));
          valid_sources = valid_sources && source_regions[bank].count != 0u;
        }
        const residency::CacheBinding &target = acquired.lease.bindings[0u];
        residency::AliasLease alias{};
        const residency::CacheKey source_key = virtual_cache_key(
            run, run.input_backing, run.input_version, epoch.failed_page - 1u);
        if (valid_sources &&
            pool.authority().issue_alias(
                acquired.lease.token,
                std::span<const residency::FrameRegion>{source_regions.data(),
                                                        source_regions.size()},
                source_regions[target_bank], source_key, target.key,
                target.frame, reuse.source_offset, reuse.target_offset,
                reuse.bytes, run.input_page_bytes, alias)) {
          requests[0u].read_offset = reuse.logical_offset;
          requests[0u].read_bytes = reuse.read_bytes;
          requests[0u].read_target_offset = reuse.read_target_offset;
          requests[0u].reuse_source_offset = reuse.source_offset;
          requests[0u].reuse_target_offset = reuse.target_offset;
          requests[0u].reuse_bytes = reuse.bytes;
          requests[0u].alias_source_frame = alias.source_frame();
          requests[0u].alias_nonce = alias.nonce();
          requests[0u].alias_source_key = alias.source_key();
          requests[0u].alias_target_key = alias.target_key();
          requests[0u].alias_source_region = alias.source_region();
          requests[0u].alias_target_region = alias.target_region();
          requests[0u].alias_frame_bytes = alias.frame_bytes();
          requests[0u].alias_owner_token = alias.owner_token();
          requests[0u].alias_generation = alias.generation();
          requests[0u].alias_frame =
              virtual_host_input_frame(run, alias.source_frame());
          requests[0u].alias_reuse = requests[0u].alias_frame != nullptr;
          if (requests[0u].alias_reuse) {
            aliases[alias_count++] = std::move(alias);
          } else {
            (void)pool.authority().release_alias(std::move(alias), true);
            requests[0u].read_offset = projected_pages[0u].logical_offset;
            requests[0u].read_bytes = projected_pages[0u].transfer_bytes;
            requests[0u].read_target_offset = projected_pages[0u].target_offset;
            requests[0u].reuse_source_offset = 0u;
            requests[0u].reuse_target_offset = 0u;
            requests[0u].reuse_bytes = 0u;
            requests[0u].alias_source_frame = 0u;
            requests[0u].alias_nonce = 0u;
            requests[0u].alias_source_key = {};
            requests[0u].alias_target_key = {};
            requests[0u].alias_source_region = {};
            requests[0u].alias_target_region = {};
            requests[0u].alias_frame_bytes = 0u;
            requests[0u].alias_owner_token = 0u;
            requests[0u].alias_generation = 0u;
          }
        }
      }
    }
  }
  if (!pool.authority().activate(acquired.lease.token)) {
    for (std::size_t index = 0u; index < alias_count; ++index) {
      (void)pool.authority().release_alias(std::move(aliases[index]), true);
    }
    return abort();
  }
  pending = prefetcher.submit(
      backing,
      std::span<const residency::PrefetchRequest>{requests.data(),
                                                  request_count},
      acquired.lease.token, speculative, coherent, coherent_deferred,
      std::span<residency::AliasLease>{aliases.data(), alias_count});
  if (!pending) {
    for (std::size_t index = 0u; index < alias_count; ++index) {
      (void)pool.authority().release_alias(std::move(aliases[index]), true);
    }
    cleanup_failed = !pool.authority().complete(acquired.lease.token, false) ||
                     cleanup_failed;
    return Status::fail(Reason::PipelineInvalid);
  }
  return Status::success();
}

} // namespace rund::compute::detail
