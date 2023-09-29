/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2022 Pascal Kuendig and Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/CG_utils.h>
#include <GPBoost/type_defs.h>
#include <GPBoost/re_comp.h>
#include <LightGBM/utils/log.h>

#include <chrono>
#include <thread> //temp
//#include <functional>
//#include <iostream>
//using namespace std;
using LightGBM::Log;

namespace GPBoost {

	void simProbeVect(RNG_t& generator, den_mat_t& Z, const bool rademacher) {

		double u;

		if (rademacher) {
			std::uniform_real_distribution<double> udist(0.0, 1.0);

			for (int i = 0; i < Z.rows(); ++i) {
				for (int j = 0; j < Z.cols(); j++) {
					u = udist(generator);
					if (u > 0.5) {
						Z(i, j) = 1.;
					}
					else {
						Z(i, j) = -1.;
					}
				}
			}
		}
		else {
			std::normal_distribution<double> ndist(0.0, 1.0);

			for (int i = 0; i < Z.rows(); ++i) {
				for (int j = 0; j < Z.cols(); j++) {
					Z(i, j) = ndist(generator);
				}
			}
		}
	}

	void LogDetStochTridiag(const std::vector<vec_t>& Tdiags,
		const  std::vector<vec_t>& Tsubdiags,
		double& ldet,
		const data_size_t num_data,
		const int t) {
		Eigen::SelfAdjointEigenSolver<den_mat_t> es;
		vec_t e1_logLambda_e1;
		
		for (int i = 0; i < t; ++i) {

			e1_logLambda_e1.setZero();

			es.computeFromTridiagonal(Tdiags[i], Tsubdiags[i]);

			e1_logLambda_e1 = es.eigenvectors().row(0).transpose().array() * es.eigenvalues().array().log() * es.eigenvectors().row(0).transpose().array();
			
			ldet += e1_logLambda_e1.sum();
		}

		ldet = ldet * num_data / t;
	} // end LogDetStochTridiag


}