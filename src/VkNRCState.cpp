//
// Created by adamyuan on 2/23/24.
//

#include "VkNRCState.hpp"

#include <array>
#include <myvk/CommandBuffer.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace nrc {

struct PackedNRCInput {
	uint32_t primitive_id, flip_bit_instance_id;
	uint32_t barycentric_2x16U;
	uint32_t scattered_dir_2x16U;
};
struct NRCEvalRecord {
	uint32_t opt1_screen_x15_y15_or_train_b2_l14_r14;
	PackedNRCInput packed_input;
};
struct NRCTrainRecord {
	float bias_r, bias_g, bias_b, factor_r, factor_g, factor_b;
	PackedNRCInput packed_input;
};
struct OptimizerState {
	uint32_t t;
	float beta1_t, beta2_t, alpha_t, alpha_t_1;
};
struct OptimizerEntry {
	float m, v, weight, ema_weight;
};
} // namespace nrc

namespace {
bool InitializeStorageLayouts(const myvk::Ptr<myvk::Queue> &queue,
                              std::initializer_list<myvk::Ptr<myvk::ImageView>> views) {
	auto pool = myvk::CommandPool::Create(queue);
	if (!pool)
		return false;
	auto command_buffer = myvk::CommandBuffer::Create(pool);
	if (!command_buffer || command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT) != VK_SUCCESS)
		return false;
	std::vector<VkImageMemoryBarrier> barriers;
	for (const auto &view : views)
		barriers.push_back(view->GetMemoryBarrier(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
		                                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL));
	command_buffer->CmdPipelineBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
	                                   {}, {}, barriers);
	if (command_buffer->End() != VK_SUCCESS)
		return false;
	auto fence = myvk::Fence::Create(queue->GetDevicePtr());
	return fence && command_buffer->Submit(fence) == VK_SUCCESS && fence->Wait() == VK_SUCCESS;
}
} // namespace

VkDeviceSize VkNRCState::GetEvalRecordBufferSize(VkExtent2D extent) {
	return (extent.width * extent.height + kTrainBatchSize * kTrainBatchCount) * sizeof(nrc::NRCEvalRecord);
}
VkDeviceSize VkNRCState::GetBatchTrainRecordBufferSize() { return kTrainBatchSize * sizeof(nrc::NRCTrainRecord); }

void VkNRCState::initialize_weights(std::span<float, kNNWeighCount> weights) {
	// (He) Kaiming Initialization
	std::normal_distribution<float> norm{0, std::sqrt(2.0f / float(kNNWidth))};
	for (auto &w : weights)
		w = norm(m_rng);
}

void VkNRCState::ResetMLPBuffers() {
	std::array<float, GetWeightCount()> fp32_weights{};
	initialize_weights(fp32_weights);
	std::array<half_float::half, GetWeightCount()> fp16_weights{};
	nrc::OptimizerState optimizer_state = {.t = 0, .beta1_t = 1.0f, .beta2_t = 1.0f, .alpha_t = 1.0f};
	std::array<nrc::OptimizerEntry, GetWeightCount()> optimizer_entries{};
	for (uint32_t i = 0; i < GetWeightCount(); ++i) {
		fp16_weights[i] = fp32_weights[i];
		optimizer_entries[i] = {
		    .weight = fp32_weights[i],
		    .ema_weight = fp32_weights[i],
		};
	}

	m_weights = myvk::Buffer::Create(GetDevicePtr(), GetWeightCount() * sizeof(uint16_t), 0,
	                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	m_use_weights = myvk::Buffer::Create(GetDevicePtr(), GetWeightCount() * sizeof(uint16_t), 0,
	                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	m_optimizer_state = myvk::Buffer::Create(GetDevicePtr(), sizeof(nrc::OptimizerState), 0,
	                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	                                             VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	m_optimizer_entries = myvk::Buffer::Create(GetDevicePtr(), GetWeightCount() * sizeof(nrc::OptimizerEntry), 0,
	                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	auto weights_staging = myvk::Buffer::CreateStaging(GetDevicePtr(), fp16_weights.begin(), fp16_weights.end());
	auto optimizer_state_staging = myvk::Buffer::CreateStaging(GetDevicePtr(), optimizer_state);
	auto optimizer_entries_staging =
	    myvk::Buffer::CreateStaging(GetDevicePtr(), optimizer_entries.begin(), optimizer_entries.end());

	auto command_buffer = myvk::CommandBuffer::Create(myvk::CommandPool::Create(m_queue_ptr));
	command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	command_buffer->CmdCopy(weights_staging, m_weights, {VkBufferCopy{.size = weights_staging->GetSize()}});
	command_buffer->CmdCopy(weights_staging, m_use_weights, {VkBufferCopy{.size = weights_staging->GetSize()}});
	command_buffer->CmdCopy(optimizer_state_staging, m_optimizer_state,
	                        {VkBufferCopy{.size = optimizer_state_staging->GetSize()}});
	command_buffer->CmdCopy(optimizer_entries_staging, m_optimizer_entries,
	                        {VkBufferCopy{.size = optimizer_entries_staging->GetSize()}});
	command_buffer->End();

	auto fence = myvk::Fence::Create(GetDevicePtr());
	command_buffer->Submit(fence);
	fence->Wait();
}

uint32_t VkNRCState::max_accumulation_window(VkExtent2D extent) const {
	constexpr VkDeviceSize kHistoryBudget = VkDeviceSize{1} << 30;
	constexpr uint32_t kMaxHistoryLayers = 64;
	const VkDeviceSize bytes_per_layer = VkDeviceSize{std::max(extent.width, 1u)} *
	                                     std::max(extent.height, 1u) * 4u * sizeof(float);
	const uint32_t budget_layers = static_cast<uint32_t>(kHistoryBudget / bytes_per_layer);
	const uint32_t device_layers =
	    GetDevicePtr()->GetPhysicalDevicePtr()->GetProperties().vk10.limits.maxImageArrayLayers;
	return std::max(1u, std::min({kMaxHistoryLayers, budget_layers, device_layers}));
}

myvk::Ptr<myvk::ImageView> VkNRCState::create_history_view(VkExtent2D extent, uint32_t window) const {
	const VkExtent2D history_extent = window == 0 ? VkExtent2D{1, 1} : extent;
	auto image = myvk::Image::CreateTexture2DArray(GetDevicePtr(), history_extent, std::max(window, 1u), 1,
	                                             VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
	return image ? myvk::ImageView::Create(image, VK_IMAGE_VIEW_TYPE_2D_ARRAY) : nullptr;
}

bool VkNRCState::SetAccumulationWindow(uint32_t window) {
	const uint32_t requested = window;
	window = std::min(window, GetMaxAccumulationWindow());
	if (window != requested)
		spdlog::warn("Accumulation window limited to {} samples at {} x {}", window,
		             m_accumulate_extent.width, m_accumulate_extent.height);
	if (window == m_accumulation_window)
		return false;
	auto history = create_history_view(m_accumulate_extent, window);
	if (!history || !InitializeStorageLayouts(m_queue_ptr, {history})) {
		spdlog::error("Cannot allocate accumulation history for {} samples; keeping {}", window,
		              m_accumulation_window);
		return false;
	}
	m_history_view = std::move(history);
	m_accumulation_window = window;
	ResetAccumulateCount();
	return true;
}

void VkNRCState::ResetAccumulateImage(VkExtent2D extent) {
	if (extent.width == 0 || extent.height == 0)
		return;
	uint32_t window = std::min(m_accumulation_window, max_accumulation_window(extent));
	if (window != m_accumulation_window)
		spdlog::warn("Resize limits accumulation window to {} samples at {} x {}", window,
		             extent.width, extent.height);
	auto image = myvk::Image::CreateTexture2D(GetDevicePtr(), extent, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
	                                          VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
	auto accumulate = image ? myvk::ImageView::Create(image, VK_IMAGE_VIEW_TYPE_2D) : nullptr;
	if (!accumulate)
		throw std::runtime_error("Cannot allocate accumulation image");
	auto history = create_history_view(extent, window);
	if (!history && window != 0) {
		spdlog::error("Cannot allocate resized accumulation history; using all-sample accumulation");
		window = 0;
		history = create_history_view(extent, window);
	}
	if (!history || !InitializeStorageLayouts(m_queue_ptr, {accumulate, history}))
		throw std::runtime_error("Cannot initialize accumulation images");
	m_accumulate_view = std::move(accumulate);
	m_history_view = std::move(history);
	m_accumulate_extent = extent;
	m_accumulation_window = window;
	ResetAccumulateCount();
}
