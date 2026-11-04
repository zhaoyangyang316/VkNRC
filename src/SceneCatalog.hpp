#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

struct SceneCamera {
	// Positions use the renderer's centered, normalized scene coordinates.
	std::array<float, 3> position{}, target{};
	float fov_degrees{60.0f};
};

struct SceneCatalogEntry {
	std::string label;
	std::string name; // Friendly scene name or OBJ stem, independent of the dropdown path.
	std::filesystem::path path;
	bool is_cornell{false};
	std::optional<SceneCamera> camera;
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
// A sibling .camera file may supply a friendly name and a normalized Y-up
// camera using name, position, target and fov lines; # introduces a comment.
// default_index identifies the bundled Cornell scene; custom_index identifies
// an explicitly supplied file. A missing/invalid custom file leaves its index
// empty so the caller can report the error instead of opening another scene.
SceneCatalog DiscoverSceneCatalog(const std::filesystem::path &argv0,
                                  const std::optional<std::filesystem::path> &custom_scene = std::nullopt);
