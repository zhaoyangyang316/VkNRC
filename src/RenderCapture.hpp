#pragma once

#include "VkNRCState.hpp"
#include <filesystem>
#include <string_view>
#include <myvk/ImageView.hpp>
#include <myvk/Queue.hpp>

// Save the accumulated linear RGB radiance, before tone mapping or GUI overlays.
// The image must contain a rendered frame, use RGBA32F with
// TRANSFER_SRC usage, and be in GENERAL layout on the supplied rendering queue.
// This synchronous operation waits for the device and restores GENERAL layout.
// PFM pixels are RGB float32, little endian, with the bottom row stored first.
bool CaptureAccumulationPFM(const myvk::Ptr<myvk::Queue> &queue,
                            const myvk::Ptr<myvk::ImageView> &image_view,
                            const std::filesystem::path &filename);

struct CaptureRenderSettings {
	bool nee{};
	VkNRCState::Method left_method{}, right_method{};
	uint32_t max_bounces{}, spp{};
	std::string_view scene_name; // UTF-8; caller keeps the name alive through the capture.
	bool fast_mode{};
	float min_roughness{VkNRCState::GetDefaultMinRoughness()};
};

struct CaptureMetadata {
	CaptureRenderSettings settings;
	bool write_header{true};
	bool print_caption{false};
};

// Same image and synchronization requirements as PFM. Saves full-precision
// RGB in an uncompressed OpenEXR scanline image, with row zero at the top.
// Frame settings can be saved in the header and/or printed on the exported pixels.
bool CaptureAccumulationEXR(const myvk::Ptr<myvk::Queue> &queue,
                            const myvk::Ptr<myvk::ImageView> &image_view,
                            const std::filesystem::path &filename, const CaptureMetadata *metadata = nullptr);
