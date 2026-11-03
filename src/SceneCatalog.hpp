#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

struct SceneCatalogEntry {
	std::string label;
	std::filesystem::path path;
	bool is_cornell{false};
};

struct SceneCatalog {
	std::vector<SceneCatalogEntry> entries;
	std::filesystem::path assets_directory;
	std::optional<size_t> default_index;
	std::optional<size_t> custom_index;
	std::string diagnostic;
};

// Find assets beside the executable, in its ancestors, or at the configured
// VKNRC_ASSET_DIRECTORY. OBJ files beneath the selected assets directory are
// listed recursively. Existing custom files are included without duplicates.
// default_index identifies the bundled Cornell scene; custom_index identifies
// an explicitly supplied file. A missing/invalid custom file leaves its index
// empty so the caller can report the error instead of opening another scene.
SceneCatalog DiscoverSceneCatalog(const std::filesystem::path &argv0,
                                  const std::optional<std::filesystem::path> &custom_scene = std::nullopt);
