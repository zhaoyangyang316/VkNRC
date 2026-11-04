//
// Created by adamyuan on 2/22/24.
//

#pragma once
#ifndef VKNRC_RG_NRCRENDERGRAPH_HPP
#define VKNRC_RG_NRCRENDERGRAPH_HPP

#include "../Camera.hpp"
#include "../VkNRCState.hpp"
#include "../VkSceneTLAS.hpp"
#include "NRCResources.hpp"
#include "SceneResources.hpp"

#include <myvk_rg/RenderGraph.hpp>

namespace rg {

class NRCRenderGraph final : public myvk_rg::RenderGraphBase {
private:
	myvk::Ptr<VkSceneTLAS> m_scene_tlas_ptr;
	myvk::Ptr<VkScene> m_scene_ptr;
	myvk::Ptr<VkNRCState> m_nrc_state_ptr;

	SceneResources create_scene_resources();
	NRCResources create_nrc_resources();

public:
	explicit NRCRenderGraph(const myvk::Ptr<VkScene> &scene, const myvk::Ptr<VkNRCState> &nrc_state_ptr,
	                        const myvk::Ptr<Camera> &camera_ptr, const myvk::Ptr<VkSceneTLAS> &scene_tlas_ptr);
	~NRCRenderGraph() final = default;
	void PreExecute() const final;
};

// Fit the linear render to the window, tone map it, then draw the live UI.
#ifdef MYVK_ENABLE_GLFW
class PresentRenderGraph final : public myvk_rg::RenderGraphBase {
public:
	PresentRenderGraph(const myvk::Ptr<myvk::FrameManager> &frame_manager,
	                   const myvk::Ptr<myvk::ImageView> &linear_image);
	~PresentRenderGraph() final = default;
};
#endif

} // namespace rg

#endif
