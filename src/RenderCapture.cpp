#include "RenderCapture.hpp"

#include <bit>
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
} // namespace

bool CaptureAccumulationPFM(const myvk::Ptr<myvk::Queue> &queue,
                            const myvk::Ptr<myvk::ImageView> &image_view,
                            const std::filesystem::path &filename) {
	static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
	static_assert(std::endian::native == std::endian::little, "PFM writer requires a little-endian host");
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
	std::ofstream output(filename, std::ios::binary);
	if (!output) {
		spdlog::error("Capture cannot open {}", filename.string());
		return false;
	}
	output << "PF\n" << extent.width << ' ' << extent.height << "\n-1.0\n";
	const auto *rgba = static_cast<const float *>(mapped_info.pMappedData);
	std::vector<float> rgb_row(size_t{extent.width} * 3);
	for (uint32_t row = extent.height; row > 0; --row) {
		const auto *source = rgba + size_t{row - 1} * extent.width * 4;
		for (uint32_t x = 0; x < extent.width; ++x) {
			rgb_row[size_t{x} * 3] = source[size_t{x} * 4];
			rgb_row[size_t{x} * 3 + 1] = source[size_t{x} * 4 + 1];
			rgb_row[size_t{x} * 3 + 2] = source[size_t{x} * 4 + 2];
		}
		// Preserve negative and nonfinite values so diagnostics can detect failures.
		output.write(reinterpret_cast<const char *>(rgb_row.data()),
		             static_cast<std::streamsize>(rgb_row.size() * sizeof(float)));
	}
	output.close();
	if (!output) {
		spdlog::error("Capture failed while writing {}", filename.string());
		return false;
	}
	spdlog::info("Captured {} x {} linear RGB radiance to {}", extent.width, extent.height, filename.string());
	return true;
}
