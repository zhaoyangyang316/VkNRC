#pragma once

#include "../FrameBenchmark.hpp"
#include "../RenderCapture.hpp"
#include <string>

// Owned by the renderer. The Android UI sends commands to the render thread.
class AndroidFrameTools {
	FrameBenchmark benchmark_;
	myvk::Ptr<myvk::ImageView> last_frame_;
	CaptureRenderSettings last_settings_;
	std::string last_scene_;

public:
	std::string StartBenchmark(unsigned seconds);
	void CancelBenchmark() { benchmark_.Cancel(); }
	bool IsBenchmarkActive() const { return benchmark_.IsActive(); }
	void ResetFrame();
	// Call after NextFrame for each submitted, presented trace. Paused redraws
	// and fence/acquire timeouts produce no new render or benchmark sample.
	std::optional<std::string> FinishRenderedFrame(const myvk::Ptr<myvk::Queue> &queue, const VkNRCState &state,
	                                               std::string_view scene_name);
	std::string Capture(const myvk::Ptr<myvk::Queue> &queue, const std::filesystem::path &directory,
	                    bool write_metadata) const;
};
