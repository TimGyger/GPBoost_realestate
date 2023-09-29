/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/utils.h>

namespace GPBoost {

	void SampleIntNoReplaceSort(int N,
		int k,
		RNG_t& gen,
		std::vector<int>& indices) {
		for (int r = N - k; r < N; ++r) {
			int v = std::uniform_int_distribution<>(0, r)(gen);
			if (std::find(indices.begin(), indices.end(), v) == indices.end()) {
				indices.push_back(v);
			}
			else {
				indices.push_back(r);
			}
		}
		std::sort(indices.begin(), indices.end());
	}//end SampleIntNoReplaceSort 

}  // namespace GPBoost
