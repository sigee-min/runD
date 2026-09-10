#pragma once

#include "../../pipeline/residency/model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace rund::compute::detail::graph_resident_plan {

struct Plan final {
  inline static constexpr std::size_t InputCount = 3u;
  inline static constexpr std::size_t StageCount = 5u;
  inline static constexpr std::size_t PageCount = 5u;
  inline static constexpr std::size_t FrameCapacity = 2u;
  inline static constexpr std::size_t BatchCount = 3u;
  inline static constexpr std::size_t ResourceCount = 8u;
  inline static constexpr std::size_t PortCount = 7u;
  inline static constexpr std::size_t InternalOwnerCount = 3u;
  inline static constexpr std::size_t TotalClassCount = 7u;

  inline static constexpr std::array<std::uint8_t, StageCount> ReadCounts{
      1u, 1u, 2u, 1u, 2u};
  inline static constexpr std::array<std::uint8_t, StageCount> WriteCounts{
      1u, 1u, 1u, 1u, 1u};
  inline static constexpr std::array<std::array<std::uint8_t, 2u>, StageCount>
      StageInputs{{{{0u, 0u}}, {{1u, 0u}}, {{3u, 4u}}, {{2u, 0u}}, {{5u, 6u}}}};
  inline static constexpr std::array<std::uint8_t, StageCount> StageOutputs{
      3u, 4u, 5u, 6u, 7u};
  inline static constexpr std::array<std::uint32_t, StageCount> OutputNext{
      2u, 2u, 4u, 4u, residency::NoGraphStage};
  inline static constexpr std::array<std::uint8_t, ResourceCount> ResourceRoles{
      0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u};
  inline static constexpr std::array<std::uint32_t, ResourceCount> FirstStage{
      0u, 1u, 3u, 0u, 1u, 2u, 3u, 4u};
  inline static constexpr std::array<std::uint32_t, ResourceCount> LastStage{
      0u, 1u, 3u, 2u, 2u, 4u, 4u, 4u};
  inline static constexpr std::array<std::uint32_t, ResourceCount>
      ProducerStage{residency::NoGraphStage,
                    residency::NoGraphStage,
                    residency::NoGraphStage,
                    0u,
                    1u,
                    2u,
                    3u,
                    residency::NoGraphStage};

  inline static constexpr std::array<std::array<std::uint8_t, 2u>, PortCount>
      Edges{{{{0u, 3u}},
             {{1u, 4u}},
             {{3u, 5u}},
             {{4u, 5u}},
             {{2u, 6u}},
             {{5u, 7u}},
             {{6u, 7u}}}};

private:
  inline static constexpr std::array<residency::GraphResourceKind,
                                     ResourceCount>
      Kinds{residency::GraphResourceKind::ExternalInput,
            residency::GraphResourceKind::ExternalInput,
            residency::GraphResourceKind::ExternalInput,
            residency::GraphResourceKind::Internal,
            residency::GraphResourceKind::Internal,
            residency::GraphResourceKind::Internal,
            residency::GraphResourceKind::Internal,
            residency::GraphResourceKind::ExternalOutput};
  inline static constexpr std::array<residency::GraphResourceRole,
                                     ResourceCount>
      Roles{residency::GraphResourceRole::Input,
            residency::GraphResourceRole::Input,
            residency::GraphResourceRole::Input,
            residency::GraphResourceRole::Intermediate,
            residency::GraphResourceRole::Intermediate,
            residency::GraphResourceRole::Intermediate,
            residency::GraphResourceRole::Intermediate,
            residency::GraphResourceRole::Output};
  inline static constexpr std::array<residency::ResourcePersistence,
                                     ResourceCount>
      Persistence{residency::ResourcePersistence::Backing,
                  residency::ResourcePersistence::Backing,
                  residency::ResourcePersistence::Backing,
                  residency::ResourcePersistence::Transient,
                  residency::ResourcePersistence::Transient,
                  residency::ResourcePersistence::Transient,
                  residency::ResourcePersistence::Transient,
                  residency::ResourcePersistence::Backing};

  [[nodiscard]] static bool shape(const residency::TiledGraphPlan &graph,
                                  const std::size_t inputs,
                                  const std::uint64_t frames,
                                  const std::uint64_t pages) noexcept {
    return inputs == InputCount && frames == FrameCapacity &&
           pages == PageCount && graph.page_count() == PageCount &&
           graph.frame_capacity() == FrameCapacity &&
           graph.batch_count() == BatchCount &&
           graph.resources().size() == ResourceCount &&
           graph.stages().size() == StageCount &&
           graph.physical_classes().size() == TotalClassCount;
  }

  [[nodiscard]] static bool
  stages(const std::span<const residency::TiledGraphStage> rows,
         std::size_t &reads, std::size_t &writes) noexcept {
    std::uint32_t prior_node = 0u;
    for (std::size_t stage = 0u; stage < rows.size(); ++stage) {
      const auto &row = rows[stage];
      if (row.domain != residency::StageDomain::Tile ||
          (stage != 0u && row.node <= prior_node) ||
          row.active_count_input != ReadCounts[stage] ||
          row.ports.size() !=
              static_cast<std::size_t>(ReadCounts[stage]) + 1u) {
        return false;
      }
      prior_node = row.node;
      for (std::size_t port = 0u; port < ReadCounts[stage]; ++port) {
        const auto &value = row.ports[port];
        if (value.resource != StageInputs[stage][port] + 1u ||
            value.access != residency::Access::Read ||
            value.program_port != port ||
            value.next_stage != residency::NoGraphStage) {
          return false;
        }
        ++reads;
      }
      const auto &output = row.ports.back();
      if (output.resource != StageOutputs[stage] + 1u ||
          output.access != residency::Access::Write ||
          output.program_port != 0u || output.next_stage != OutputNext[stage]) {
        return false;
      }
      ++writes;
    }
    return reads == PortCount && writes == StageCount;
  }

  [[nodiscard]] static bool
  rows(const std::span<const residency::TiledGraphResource> resources,
       const std::uint64_t pages, const std::uint64_t page_bytes,
       const std::uint64_t logical_bytes, const FixedFormat format) noexcept {
    if (page_bytes == 0u ||
        pages > std::numeric_limits<std::uint64_t>::max() / page_bytes) {
      return false;
    }
    const std::uint64_t materialized_bytes = pages * page_bytes;
    for (std::size_t index = 0u; index < resources.size(); ++index) {
      const auto &row = resources[index];
      const std::uint64_t expected_bytes =
          Kinds[index] == residency::GraphResourceKind::Internal &&
                  Persistence[index] ==
                      residency::ResourcePersistence::Transient
              ? materialized_bytes
              : logical_bytes;
      if (row.resource != index + 1u || row.type != Type::U64 ||
          row.format != format || row.page_bytes != page_bytes ||
          row.logical_bytes != expected_bytes || row.kind != Kinds[index] ||
          row.role != Roles[index] || row.persistence != Persistence[index] ||
          row.first_stage != FirstStage[index] ||
          row.last_stage != LastStage[index] ||
          row.producer_stage != ProducerStage[index] || row.physical_id == 0u) {
        return false;
      }
    }
    const auto &x = resources[3u];
    const auto &y = resources[4u];
    const auto &z = resources[5u];
    const auto &w = resources[6u];
    const auto &output = resources[7u];
    if (x.physical_id != w.physical_id || x.color != w.color ||
        x.physical_id == y.physical_id || x.physical_id == z.physical_id ||
        y.physical_id == z.physical_id || output.physical_id == x.physical_id ||
        output.physical_id == y.physical_id ||
        output.physical_id == z.physical_id ||
        output.physical_id == resources[0u].physical_id ||
        output.physical_id == resources[1u].physical_id ||
        output.physical_id == resources[2u].physical_id ||
        resources[0u].physical_id == resources[1u].physical_id ||
        resources[0u].physical_id == resources[2u].physical_id ||
        resources[1u].physical_id == resources[2u].physical_id ||
        resources[0u].color == resources[1u].color ||
        resources[0u].color == resources[2u].color ||
        resources[1u].color == resources[2u].color || x.color == y.color ||
        x.color == z.color || y.color == z.color) {
      return false;
    }
    return true;
  }

  [[nodiscard]] static bool
  owners(const std::span<const residency::TiledGraphResource> resources,
         const std::span<const residency::TiledGraphPhysicalClass> classes,
         const std::uint64_t page_bytes, const FixedFormat format) noexcept {
    std::size_t inputs = 0u;
    std::size_t internal = 0u;
    std::size_t outputs = 0u;
    for (std::size_t index = 0u; index < classes.size(); ++index) {
      const auto &owner = classes[index];
      if (owner.physical_id == 0u || owner.type != Type::U64 ||
          owner.format != format || owner.page_bytes != page_bytes) {
        return false;
      }
      for (std::size_t prior = 0u; prior < index; ++prior) {
        if (classes[prior].physical_id == owner.physical_id) {
          return false;
        }
      }
      bool referenced = false;
      for (const auto &resource : resources) {
        if (resource.physical_id != owner.physical_id) {
          continue;
        }
        if (resource.role != owner.role || resource.color != owner.color) {
          return false;
        }
        referenced = true;
      }
      if (!referenced) {
        return false;
      }
      if (owner.role == residency::GraphResourceRole::Input) {
        ++inputs;
      } else if (owner.role == residency::GraphResourceRole::Intermediate) {
        ++internal;
      } else if (owner.role == residency::GraphResourceRole::Output) {
        ++outputs;
      } else {
        return false;
      }
    }
    for (const auto &resource : resources) {
      bool found = false;
      for (const auto &owner : classes) {
        found = found || resource.physical_id == owner.physical_id;
      }
      if (!found) {
        return false;
      }
    }
    return inputs == InputCount && internal == InternalOwnerCount &&
           outputs == 1u;
  }

public:
  [[nodiscard]] static bool
  matches(const residency::TiledGraphPlan &graph, const std::size_t inputs,
          const std::uint64_t frames, const std::uint64_t pages,
          const std::uint64_t page_bytes,
          const std::uint64_t logical_bytes) noexcept {
    const auto resources = graph.resources();
    if (resources.empty() || !shape(graph, inputs, frames, pages) ||
        page_bytes == 0u || logical_bytes == 0u ||
        !rows(resources, pages, page_bytes, logical_bytes,
              resources.front().format)) {
      return false;
    }
    std::size_t reads = 0u;
    std::size_t writes = 0u;
    return stages(graph.stages(), reads, writes) &&
           owners(resources, graph.physical_classes(), page_bytes,
                  resources.front().format);
  }
};

} // namespace rund::compute::detail::graph_resident_plan
