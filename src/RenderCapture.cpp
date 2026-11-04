#include "RenderCapture.hpp"
#include "CaptureCaption.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>
#include <vector>

#include <myvk/CommandBuffer.hpp>
#include <spdlog/spdlog.h>

namespace {
// Keep the allocation accessible so noncoherent host memory can be invalidated.
struct ReadbackBuffer {
	VmaAllocator allocator{};
	VkBuffer buffer{VK_NULL_HANDLE};
	VmaAllocation allocation{VK_NULL_HANDLE};

	~ReadbackBuffer() {
		if (buffer != VK_NULL_HANDLE)
			vmaDestroyBuffer(allocator, buffer, allocation);
	}
};

bool CheckResult(VkResult result, const char *operation) {
	if (result == VK_SUCCESS)
		return true;
	spdlog::error("Capture: {} failed (VkResult {})", operation, static_cast<int>(result));
	return false;
}
enum class Format { PFM, EXR };

// Both formats below store IEEE binary32 in little-endian byte order.
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::endian::native == std::endian::little);

template <typename T> void WriteValue(std::ostream &output, const T &value) {
	output.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void WriteAttributeHeader(std::ostream &output, std::string_view name, const char *type, int32_t bytes) {
	output << name << '\0' << type << '\0';
	WriteValue(output, bytes);
}

void WriteStringAttribute(std::ostream &output, std::string_view name, std::string_view value) {
	if (value.size() > std::numeric_limits<int32_t>::max()) {
		output.setstate(std::ios::failbit);
		return;
	}
	const auto bytes = static_cast<int32_t>(value.size());
	WriteAttributeHeader(output, name, "string", bytes);
	if (bytes != 0)
		output.write(value.data(), bytes);
}

void WriteCaptureMetadata(std::ostream &output, const CaptureRenderSettings &settings, VkExtent3D extent) {
	const auto left = VkNRCState::MethodName(settings.left_method), right = VkNRCState::MethodName(settings.right_method);
	constexpr auto int_max = std::numeric_limits<int32_t>::max();
	if (left.empty() || right.empty() || settings.max_bounces > int_max) {
		output.setstate(std::ios::failbit);
		return;
	}
	WriteStringAttribute(output, "vknrc.resolution", fmt::format("{}x{}", extent.width, extent.height));
	WriteStringAttribute(output, "vknrc.nee", settings.nee ? "on" : "off");
	WriteStringAttribute(output, "vknrc.fastMode", settings.fast_mode ? "on" : "off");
	WriteAttributeHeader(output, "vknrc.minRoughness", "float", 4);
	WriteValue(output, settings.min_roughness);
	WriteAttributeHeader(output, "vknrc.maxBounces", "int", 4);
	WriteValue(output, static_cast<int32_t>(settings.max_bounces));
	if (settings.spp <= int_max) {
		WriteAttributeHeader(output, "vknrc.spp", "int", 4);
		WriteValue(output, static_cast<int32_t>(settings.spp));
	} else {
		// OpenEXR has no standard uint attribute. Binary64 exactly represents every uint32_t.
		static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
		WriteAttributeHeader(output, "vknrc.spp", "double", 8);
		WriteValue(output, static_cast<double>(settings.spp));
	}
	WriteStringAttribute(output, "vknrc.left.method", left);
	WriteStringAttribute(output, "vknrc.right.method", right);
	WriteStringAttribute(output, "vknrc.scene", settings.scene_name);
}

// Minimal single-part, uncompressed scanline layout from
// https://openexr.com/en/latest/OpenEXRFileLayout.html
void WriteEXR(std::ostream &output, VkExtent3D extent, const float *rgba, const CaptureMetadata *metadata) {
	capture_detail::Caption caption;
	if (metadata && metadata->print_caption && !capture_detail::MakeCaption(caption, extent, *metadata)) {
		output.setstate(std::ios::failbit);
		return;
	}
	WriteValue(output, uint32_t{20000630}); // OpenEXR magic number
	WriteValue(output, uint32_t{2});        // File version, no flags
	// Each one-letter channel name takes two bytes, followed by four 32-bit fields.
	WriteAttributeHeader(output, "channels", "chlist", 3 * (2 + 4 * 4) + 1);
	for (char name : {'B', 'G', 'R'}) {
		output << name << '\0';
		WriteValue(output, int32_t{2}); // FLOAT
		WriteValue(output, uint32_t{0}); // pLinear and three reserved bytes
		WriteValue(output, int32_t{1}); // xSampling
		WriteValue(output, int32_t{1}); // ySampling
	}
	output.put('\0');
	WriteAttributeHeader(output, "compression", "compression", 1);
	output.put('\0'); // NO_COMPRESSION
	const std::array<int32_t, 4> window{0, 0, static_cast<int32_t>(extent.width - 1),
	                                    static_cast<int32_t>(extent.height - 1)};
	for (const char *name : {"dataWindow", "displayWindow"}) {
		WriteAttributeHeader(output, name, "box2i", 16);
		WriteValue(output, window);
	}
	WriteAttributeHeader(output, "lineOrder", "lineOrder", 1);
	output.put('\0'); // INCREASING_Y
	WriteAttributeHeader(output, "pixelAspectRatio", "float", 4);
	WriteValue(output, 1.0f);
	WriteAttributeHeader(output, "screenWindowCenter", "v2f", 8);
	WriteValue(output, std::array<float, 2>{0.0f, 0.0f});
	WriteAttributeHeader(output, "screenWindowWidth", "float", 4);
	WriteValue(output, 1.0f);
	if (metadata && metadata->write_header)
		WriteCaptureMetadata(output, metadata->settings, extent);
	output.put('\0'); // End of header

	const auto header_bytes = static_cast<std::streamoff>(output.tellp());
	if (header_bytes < 0) {
		output.setstate(std::ios::failbit);
		return;
	}
	// Each scanline chunk has a signed 32-bit y, byte count, and planar B/G/R data.
	const int32_t row_bytes = static_cast<int32_t>(size_t{extent.width} * 3 * sizeof(float));
	uint64_t offset = static_cast<uint64_t>(header_bytes) + uint64_t{extent.height} * sizeof(uint64_t);
	for (uint32_t y = 0; y < extent.height; ++y) {
		WriteValue(output, offset);
		offset += 8u + static_cast<uint64_t>(row_bytes);
	}
	std::vector<float> row(size_t{extent.width} * 3);
	for (uint32_t y = 0; y < extent.height && output; ++y) {
		WriteValue(output, static_cast<int32_t>(y));
		WriteValue(output, row_bytes);
		const float *source = rgba + size_t{y} * extent.width * 4;
		for (size_t x = 0; x < extent.width; ++x) {
			row[x] = source[4 * x + 2];
			row[extent.width + x] = source[4 * x + 1];
			row[2 * size_t{extent.width} + x] = source[4 * x];
		}
		caption.CompositeRow(y, extent.width, row.data());
		output.write(reinterpret_cast<const char *>(row.data()), row_bytes);
	}
}

void WritePFM(std::ostream &output, VkExtent3D extent, const float *rgba) {
	output << "PF\n" << extent.width << ' ' << extent.height << "\n-1.0\n";
	std::vector<float> rgb_row(size_t{extent.width} * 3);
	for (uint32_t row = extent.height; row > 0 && output; --row) {
		const auto *source = rgba + size_t{row - 1} * extent.width * 4;
		for (size_t x = 0; x < extent.width; ++x) {
			rgb_row[x * 3] = source[x * 4];
			rgb_row[x * 3 + 1] = source[x * 4 + 1];
			rgb_row[x * 3 + 2] = source[x * 4 + 2];
		}
		output.write(reinterpret_cast<const char *>(rgb_row.data()),
		             static_cast<std::streamsize>(rgb_row.size() * sizeof(float)));
	}
}

bool CaptureAccumulation(const myvk::Ptr<myvk::Queue> &queue, const myvk::Ptr<myvk::ImageView> &image_view,
                         const std::filesystem::path &filename, Format format, const CaptureMetadata *metadata = nullptr) {
	if (!queue || !image_view || filename.empty()) {
		spdlog::error("Capture requires a rendering queue, image view, and output filename");
		return false;
	}
	const auto &image = image_view->GetImagePtr();
	const auto &device = queue->GetDevicePtr();
	const auto extent = image->GetExtent();
	const auto range = image_view->GetSubresourceRange();
	if (image->GetDevicePtr() != device || image->GetFormat() != VK_FORMAT_R32G32B32A32_SFLOAT ||
	    !(image->GetUsage() & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) || extent.width == 0 || extent.height == 0 ||
	    extent.depth != 1 || image->GetArrayLayers() != 1 || range.baseMipLevel != 0 || range.levelCount != 1 ||
	    range.baseArrayLayer != 0 || range.layerCount != 1 || range.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT) {
		spdlog::error("Capture requires a single-layer RGBA32F image with TRANSFER_SRC usage on the rendering device");
		return false;
	}
	constexpr auto kMaxExrInt = std::numeric_limits<int32_t>::max();
	if (format == Format::EXR && (extent.width > kMaxExrInt / (3 * sizeof(float)) || extent.height > kMaxExrInt)) {
		spdlog::error("Capture extent exceeds OpenEXR scanline limits");
		return false;
	}
	if (!CheckResult(device->WaitIdle(), "device wait"))
		return false;

	const VkDeviceSize byte_count = VkDeviceSize{extent.width} * extent.height * 4 * sizeof(float);
	ReadbackBuffer readback{.allocator = device->GetAllocatorHandle()};
	VkBufferCreateInfo buffer_info{
	    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
	    .size = byte_count,
	    .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VmaAllocationCreateInfo allocation_info{
	    .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
	    .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
	};
	VmaAllocationInfo mapped_info{};
	if (!CheckResult(vmaCreateBuffer(readback.allocator, &buffer_info, &allocation_info, &readback.buffer,
	                                 &readback.allocation, &mapped_info),
	                 "readback buffer allocation"))
		return false;
	if (!mapped_info.pMappedData) {
		spdlog::error("Capture readback allocation is not mapped");
		return false;
	}

	auto command_pool = myvk::CommandPool::Create(queue, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
	if (!command_pool) {
		spdlog::error("Capture command pool creation failed");
		return false;
	}
	auto command_buffer = myvk::CommandBuffer::Create(command_pool);
	auto fence = myvk::Fence::Create(device);
	if (!command_buffer || !fence) {
		spdlog::error("Capture command buffer or fence creation failed");
		return false;
	}
	if (!CheckResult(command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT), "command buffer begin"))
		return false;
	command_buffer->CmdPipelineBarrier(
	    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, {}, {},
	    {image->GetMemoryBarrier(range, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
	                             VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)});
	VkBufferImageCopy copy{
	    .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
	    .imageExtent = extent,
	};
	vkCmdCopyImageToBuffer(command_buffer->GetHandle(), image->GetHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                       readback.buffer, 1, &copy);
	VkBufferMemoryBarrier host_barrier{
	    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
	    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
	    .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .buffer = readback.buffer,
	    .offset = 0,
	    .size = byte_count,
	};
	command_buffer->CmdPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, {}, {host_barrier}, {});
	command_buffer->CmdPipelineBarrier(
	    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, {}, {},
	    {image->GetMemoryBarrier(range, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
	                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL)});
	if (!CheckResult(command_buffer->End(), "command buffer end") ||
	    !CheckResult(command_buffer->Submit(fence), "readback submission") || !CheckResult(fence->Wait(), "readback wait") ||
	    !CheckResult(vmaInvalidateAllocation(readback.allocator, readback.allocation, 0, byte_count), "host invalidation"))
		return false;

	std::error_code directory_error;
	if (!filename.parent_path().empty())
		std::filesystem::create_directories(filename.parent_path(), directory_error);
	if (directory_error) {
		spdlog::error("Capture cannot create output directory: {}", directory_error.message());
		return false;
	}
	const auto utf8 = filename.u8string();
	const std::string name(utf8.begin(), utf8.end());
	const bool new_exr = format == Format::EXR && !std::filesystem::exists(filename, directory_error) && !directory_error;
	std::ofstream output(filename, std::ios::binary);
	if (!output) {
		spdlog::error("Capture cannot open {}", name);
		return false;
	}
	const auto *rgba = static_cast<const float *>(mapped_info.pMappedData);
	if (format == Format::EXR)
		WriteEXR(output, extent, rgba, metadata);
	else
		WritePFM(output, extent, rgba);
	output.close();
	if (!output) {
		spdlog::error("Capture failed while writing {}", name);
		if (new_exr) {
			std::error_code cleanup_error;
			std::filesystem::remove(filename, cleanup_error);
			if (cleanup_error)
				spdlog::warn("Cannot remove incomplete capture {}: {}", name, cleanup_error.message());
		}
		return false;
	}
	spdlog::info("Captured {} x {} linear RGB radiance to {}", extent.width, extent.height, name);
	return true;
}
} // namespace

bool CaptureAccumulationPFM(const myvk::Ptr<myvk::Queue> &queue, const myvk::Ptr<myvk::ImageView> &image_view,
                            const std::filesystem::path &filename) {
	return CaptureAccumulation(queue, image_view, filename, Format::PFM);
}

bool CaptureAccumulationEXR(const myvk::Ptr<myvk::Queue> &queue, const myvk::Ptr<myvk::ImageView> &image_view,
                            const std::filesystem::path &filename, const CaptureMetadata *metadata) {
	return CaptureAccumulation(queue, image_view, filename, Format::EXR, metadata);
}
