#include "SceneCatalog.hpp"

#include <algorithm>
#include <cctype>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace {
namespace fs = std::filesystem;

std::string Utf8(const fs::path &path) {
	const auto text = path.generic_u8string();
	return {reinterpret_cast<const char *>(text.data()), text.size()};
}

fs::path NormalizePath(const fs::path &path) {
	std::error_code error;
	auto normalized = fs::weakly_canonical(path, error);
	if (!error)
		return normalized;
	error.clear();
	normalized = fs::absolute(path, error);
	return error ? path.lexically_normal() : normalized.lexically_normal();
}

bool SameFile(const fs::path &left, const fs::path &right) {
	std::error_code error;
	if (fs::equivalent(left, right, error))
		return true;
	return left == right;
}

bool IsOBJ(const fs::path &path) {
	auto extension = Utf8(path.extension());
	std::transform(extension.begin(), extension.end(), extension.begin(),
	               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return extension == ".obj";
}

fs::path ExecutablePath(const fs::path &argv0) {
#ifdef _WIN32
	// argv[0] can be relative or just a command name even when launched elsewhere.
	std::vector<wchar_t> buffer(512);
	while (buffer.size() <= 32768) {
		const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
		if (length == 0)
			break;
		if (length < buffer.size())
			return NormalizePath(fs::path(std::wstring(buffer.data(), length)));
		buffer.resize(buffer.size() * 2);
	}
#elif defined(__linux__)
	std::error_code error;
	auto executable = fs::read_symlink("/proc/self/exe", error);
	if (!error)
		return NormalizePath(executable);
#endif
	// Other platforms and unusual process hosts can still supply a full argv[0].
	return NormalizePath(argv0);
}

void AddDiagnostic(SceneCatalog &catalog, const std::string &message) {
	if (!catalog.diagnostic.empty())
		catalog.diagnostic += '\n';
	catalog.diagnostic += message;
}

void AddEntry(SceneCatalog &catalog, const fs::path &path, const fs::path &cornell_path, bool custom) {
	const auto normalized = NormalizePath(path);
	for (const auto &entry : catalog.entries)
		if (SameFile(entry.path, normalized))
			return;
	const bool is_cornell = !cornell_path.empty() && SameFile(normalized, cornell_path);
	std::string label;
	if (is_cornell) {
		label = "Cornell Box";
	} else if (!custom) {
		label = Utf8(normalized.lexically_relative(catalog.assets_directory));
	} else {
		label = Utf8(normalized) + " (custom)";
	}
	catalog.entries.push_back({.label = std::move(label), .path = normalized, .is_cornell = is_cornell});
}
} // namespace

SceneCatalog DiscoverSceneCatalog(const std::filesystem::path &argv0,
                                  const std::optional<std::filesystem::path> &custom_scene) {
	SceneCatalog catalog;
	const fs::path executable_directory = ExecutablePath(argv0).parent_path();
	std::vector<fs::path> candidates;
	for (auto directory = executable_directory; !directory.empty();) {
		candidates.push_back(directory / "assets");
		const auto parent = directory.parent_path();
		if (parent == directory)
			break;
		directory = parent;
	}
#ifdef VKNRC_ASSET_DIRECTORY
	candidates.emplace_back(VKNRC_ASSET_DIRECTORY);
#endif
	for (const auto &candidate : candidates) {
		std::error_code error;
		if (fs::is_directory(candidate, error)) {
			catalog.assets_directory = NormalizePath(candidate);
			break;
		}
	}

	fs::path cornell_path;
	if (!catalog.assets_directory.empty()) {
		cornell_path = catalog.assets_directory / "cornell-box" / "cornell-box.obj";
		std::error_code error;
		fs::recursive_directory_iterator iterator(catalog.assets_directory, fs::directory_options::skip_permission_denied,
		                                          error),
		    end;
		if (error)
			AddDiagnostic(catalog, "Cannot scan " + Utf8(catalog.assets_directory) + ": " + error.message());
		while (!error && iterator != end) {
			std::error_code file_error;
			if (IsOBJ(iterator->path()) && iterator->is_regular_file(file_error))
				AddEntry(catalog, iterator->path(), cornell_path, false);
			iterator.increment(error);
			if (error)
				AddDiagnostic(catalog, "Asset scan stopped in " + Utf8(catalog.assets_directory) + ": " + error.message());
		}
	} else {
		AddDiagnostic(catalog, "No assets directory found beside " + Utf8(executable_directory) +
		                           ", in its parent folders, or at the configured asset directory. "
		                           "Restore assets/cornell-box/cornell-box.obj or supply an OBJ path.");
	}

	std::optional<fs::path> valid_custom;
	if (custom_scene) {
		std::error_code error;
		if (!custom_scene->empty() && IsOBJ(*custom_scene) && fs::is_regular_file(*custom_scene, error)) {
			valid_custom = NormalizePath(*custom_scene);
			AddEntry(catalog, *valid_custom, cornell_path, true);
		} else {
			AddDiagnostic(catalog, "Custom scene is not an existing OBJ file: " + Utf8(*custom_scene) +
			                           (error ? " (" + error.message() + ")" : ""));
		}
	}
	std::sort(catalog.entries.begin(), catalog.entries.end(), [](const auto &left, const auto &right) {
		if (left.is_cornell != right.is_cornell)
			return left.is_cornell;
		return left.label < right.label;
	});
	for (size_t i = 0; i < catalog.entries.size(); ++i) {
		if (catalog.entries[i].is_cornell)
			catalog.default_index = i;
		if (valid_custom && SameFile(catalog.entries[i].path, *valid_custom))
			catalog.custom_index = i;
	}
	if (!catalog.default_index && !catalog.assets_directory.empty())
		AddDiagnostic(catalog, "Default Cornell scene was not found at " + Utf8(cornell_path) +
		                           ". Restore the bundled scene or supply an OBJ path.");
	return catalog;
}
