//
// Created by adamyuan on 2/23/24.
//

#pragma once
#ifndef VKNRC_VKNRCSTATE_HPP
#define VKNRC_VKNRCSTATE_HPP

#include <algorithm>
#include <glm/glm.hpp>
#include <half.hpp>
#include <myvk/Buffer.hpp>
#include <myvk/Image.hpp>
#include <myvk/ImageView.hpp>
#include <random>
#include <span>

class VkNRCState final : public myvk::DeviceObjectBase {
public:
	enum Method { kNone, kNRC, kCache };

private:
	inline static constexpr float kDefaultTrainProbability = 0.03f;
	inline static constexpr uint32_t kDefaultMaxBounces = 64, kMaxSupportedBounces = 64;
	inline static constexpr uint32_t kNNHiddenLayers = 5, kNNWidth = 64, kNNOutWidth = 3, kTrainBatchSize = 16384,
	                                 kTrainBatchCount = 4;
	inline static constexpr uint32_t kNNWeighCount = kNNWidth * kNNWidth * kNNHiddenLayers + kNNWidth * kNNOutWidth;

	myvk::Ptr<myvk::Queue> m_queue_ptr;
	// use_weights: weights used by actual renderer
	myvk::Ptr<myvk::Buffer> m_weights, m_use_weights, m_optimizer_state, m_optimizer_entries;
	myvk::Ptr<myvk::ImageView> m_accumulate_view, m_history_view;
	VkExtent2D m_accumulate_extent{};
	uint32_t m_seed{};
	std::mt19937 m_rng{std::random_device{}()};
	Method m_left_method{kNRC}, m_right_method{kNRC};
	bool m_accumulate{false};
	uint32_t m_accumulate_count{0};
	uint32_t m_accumulation_window{0}, m_history_write_index{0};
	bool m_use_ema_weights{false};
	bool m_use_nee{true};
	bool m_indirect_only{false};
	float m_train_probability{kDefaultTrainProbability};
	uint32_t m_max_bounces{kDefaultMaxBounces};

	void initialize_weights(std::span<float, kNNWeighCount> weights);
	uint32_t max_accumulation_window(VkExtent2D extent) const;
	myvk::Ptr<myvk::ImageView> create_history_view(VkExtent2D extent, uint32_t window) const;

public:
	inline VkNRCState(const myvk::Ptr<myvk::Queue> &queue_ptr, VkExtent2D extent) : m_queue_ptr(queue_ptr) {
		ResetAccumulateImage(extent);
		ResetMLPBuffers();
	}
	inline ~VkNRCState() final = default;

	inline const auto &GetAccumulateImageView() const { return m_accumulate_view; }
	inline const auto &GetHistoryImageView() const { return m_history_view; }
	inline const auto &GetWeightBuffer() const { return m_weights; }
	inline const auto &GetUseWeightBuffer() const { return m_use_weights; }
	inline const auto &GetOptimizerEntryBuffer() const { return m_optimizer_entries; }
	inline const auto &GetOptimizerStateBuffer() const { return m_optimizer_state; }

	inline Method GetLeftMethod() const { return m_left_method; }
	inline Method GetRightMethod() const { return m_right_method; }
	inline bool IsAccumulate() const { return m_accumulate; }
	inline uint32_t GetAccumulateCount() const { return m_accumulate_count; }
	// Sample count represented by the image after NextFrame().
	inline uint32_t GetRenderedSPP() const { return m_accumulate ? m_accumulate_count : 1u; }
	inline uint32_t GetAccumulationWindow() const { return m_accumulation_window; }
	inline uint32_t GetHistoryWriteIndex() const { return m_history_write_index; }
	inline uint32_t GetMaxAccumulationWindow() const { return max_accumulation_window(m_accumulate_extent); }
	inline bool IsUseEMAWeights() const { return m_use_ema_weights; }
	inline bool IsUseNEE() const { return m_use_nee; }
	inline bool IsIndirectOnly() const { return m_indirect_only; }
	inline float GetTrainProbability() const { return m_train_probability; }
	inline uint32_t GetMaxBounces() const { return m_max_bounces; }

	inline void SetLeftMethod(Method method) { m_left_method = method; }
	inline void SetRightMethod(Method method) { m_right_method = method; }
	inline void SetAccumulate(bool accumulate) {
		m_accumulate = accumulate;
		if (!accumulate)
			ResetAccumulateCount();
	}
	inline void ResetAccumulateCount() {
		m_accumulate_count = 0;
		m_history_write_index = 0;
	}
	// Zero averages all accumulated samples; a positive value keeps exactly the
	// last N samples. The caller waits for in-flight rendering before reallocating.
	bool SetAccumulationWindow(uint32_t window);
	inline void SetUseEMAWeights(bool use_ema_weights) { m_use_ema_weights = use_ema_weights; }
	// Switching the sampling estimator changes the training targets; the caller
	// resets the MLP after this reports a change.
	inline bool SetUseNEE(bool use_nee) {
		if (m_use_nee == use_nee)
			return false;
		m_use_nee = use_nee;
		ResetAccumulateCount();
		return true;
	}
	// This selects the displayed lighting contribution without changing cache targets.
	inline bool SetIndirectOnly(bool indirect_only) {
		if (m_indirect_only == indirect_only)
			return false;
		m_indirect_only = indirect_only;
		ResetAccumulateCount();
		return true;
	}
	inline void SetTrainProbability(float train_probability) { m_train_probability = train_probability; }
	// The caller also resets the MLP when changing depth, since training targets
	// depend on the trace limit. The cache itself is not conditioned on depth.
	inline bool SetMaxBounces(uint32_t max_bounces) {
		max_bounces = std::clamp(max_bounces, 1u, kMaxSupportedBounces);
		if (m_max_bounces == max_bounces)
			return false;
		m_max_bounces = max_bounces;
		ResetAccumulateCount();
		return true;
	}

	inline uint32_t GetSeed() const { return m_seed; }
	inline void SetRandomSeed(uint32_t seed) { m_rng.seed(seed); m_seed = seed; }

	inline void NextFrame(bool rendered = true) {
		if (!rendered)
			return;
		if (m_accumulate) {
			if (m_accumulation_window != 0) {
				m_accumulate_count = std::min(m_accumulate_count + 1u, m_accumulation_window);
				m_history_write_index = (m_history_write_index + 1u) % m_accumulation_window;
			} else
				++m_accumulate_count;
		}
		m_seed = std::uniform_int_distribution<uint32_t>{0}(m_rng);
	}

	void ResetAccumulateImage(VkExtent2D extent);
	void ResetMLPBuffers();

	inline const myvk::Ptr<myvk::Device> &GetDevicePtr() const { return m_queue_ptr->GetDevicePtr(); }
	static VkDeviceSize GetEvalRecordBufferSize(VkExtent2D extent);
	static VkDeviceSize GetBatchTrainRecordBufferSize();
	static constexpr uint32_t GetTrainBatchCount() { return kTrainBatchCount; }
	static constexpr uint32_t GetTrainBatchSize() { return kTrainBatchSize; }
	static constexpr uint32_t GetWeightCount() { return kNNWeighCount; }
	static constexpr float GetDefaultTrainProbability() { return kDefaultTrainProbability; }
	static constexpr uint32_t GetDefaultMaxBounces() { return kDefaultMaxBounces; }
	static constexpr uint32_t GetMaxSupportedBounces() { return kMaxSupportedBounces; }
};

#endif // VKNRC_VKNRCSTATE_HPP
