//
// Created by adamyuan on 2/22/24.
//

#pragma once
#ifndef VKNRC_RG_SCENERESOURCES_HPP
#define VKNRC_RG_SCENERESOURCES_HPP

#include <myvk_rg/RenderGraph.hpp>

namespace rg {

struct SceneResources {
	myvk_rg::Buffer tlas, vertices, vertex_indices, texcoords, texcoord_indices, materials, material_ids, transforms;
	myvk_rg::Buffer lights, light_membership;
	std::vector<myvk_rg::Image> textures;
	myvk::Ptr<myvk::Sampler> texture_sampler;
};

} // namespace rg

#endif
