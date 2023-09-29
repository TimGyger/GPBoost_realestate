/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2022 Tim Gyger and Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_CG_UTILS_
#define GPB_CG_UTILS_

//#include <functional>
//#include <iostream>
#include <GPBoost/type_defs.h>
#include <GPBoost/re_comp.h>

#include <LightGBM/utils/log.h>
#include <chrono>
#include <thread> //temp

//#include <GPBoost/likelihoods.h>
//#include <cmath>
//using namespace std;
using LightGBM::Log;

namespace GPBoost {
	/*!
	* \brief Preconditioned conjugate gradient descent to solve Au=rhs when rhs is a vector
	*		 A=(C_s + C_nm*(C_m)^(-1)*C_mn) is a symmetric matrix of dimension nxn and a Full Scale Approximation for Sigma^-1
	*		 P = diag(C_s) + C_nm*(C_m)^(-1)*C_mn or P = mean(diag(C_s)) + C_nm*(C_m)^(-1)*C_mn is used as preconditioner.
	* \param sigma_resid Residual Matrix C_s
	* \param sigma_cross_cov Matrix C_mn in Predictive Process Part C_nm*(C_m)^(-1)*C_mn
	* \param sigma_ip Matrix C_m in Predictive Process Part C_nm*(C_m)^(-1)*C_mn
	* \param chol_ip_cross_cov Cholesky Factor of C_m, the inducing point matrix, times cross-covariance
	* \param rhs Vector of dimension nx1 on the rhs
	* \param[out] u Approximative solution of the linear system (solution written on input) (must have been declared with the correct n-dimension)
	* \param[out] NaN_found Is set to true, if NaN is found in the residual of conjugate gradient algorithm
	* \param p Number of conjugate gradient steps
	* \param warm_start If false, u is set to zero at the beginning of the algorithm
	* \param delta_conv tolerance for checking convergence
	* \param THRESHOLD_ZERO_RHS_CG If the L1-norm of the rhs is below this threshold the CG is not executed and a vector u of 0's is returned
	* \param cg_preconditioner_type Type of preconditoner used for the conjugate gradient algorithm
	*/
	template <class T_mat>
	void CGFSA(const T_mat& sigma_resid,
		const den_mat_t& sigma_cross_cov,
		const den_mat_t& chol_ip_cross_cov,
		const vec_t& rhs,
		vec_t& u,
		bool& NaN_found,
		int p,
		const double delta_conv,
		const double THRESHOLD_ZERO_RHS_CG,
		const string_t cg_preconditioner_type,
		const chol_den_mat_t& chol_fact_woodbury_preconditioner,
		const vec_t& diagonal_approx_inv_preconditioner) {

		p = std::min(p, (int)rhs.size());

		vec_t r, r_old;
		vec_t z, z_old;
		vec_t h;
		vec_t v;

		vec_t diag_sigma_resid_inv_r, sigma_cross_cov_diag_sigma_resid_inv_r, mean_diag_sigma_resid_inv_r, sigma_cross_cov_mean_diag_sigma_resid_inv_r;
		bool early_stop_alg = false;
		double a = 0;
		double b = 1;
		double r_squared_norm;

		//Avoid numerical instabilites when rhs is de facto 0
		if (rhs.cwiseAbs().sum() < THRESHOLD_ZERO_RHS_CG) {
			//Log::REInfo("0 - return 0");
			u.setZero();
			return;
		}
		Log::REInfo("%g", delta_conv);
		bool is_zero = u.isZero(0);

		if (is_zero) {
			r = rhs;
		}
		else {
			r = rhs - sigma_resid * u - (chol_ip_cross_cov.transpose() * (chol_ip_cross_cov * u));//r = rhs - A * u
		}

		//z = P^(-1) r
		if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
			//D^-1*r
			diag_sigma_resid_inv_r = diagonal_approx_inv_preconditioner.asDiagonal() * r; // ??? cwiseProd (TODO)
			
			//Cmn*D^-1*r
			sigma_cross_cov_diag_sigma_resid_inv_r = sigma_cross_cov.transpose() * diag_sigma_resid_inv_r;
			//P^-1*r using Woodbury Identity
			z = diag_sigma_resid_inv_r - (diagonal_approx_inv_preconditioner.asDiagonal() * (sigma_cross_cov * chol_fact_woodbury_preconditioner.solve(sigma_cross_cov_diag_sigma_resid_inv_r)));
			
		}
		else if (cg_preconditioner_type == "none") {
			z = r;
		}
		else {
			Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
		}
		h = z;
		
		for (int j = 0; j < p; ++j) {
			//The following matrix-vector operation is the expensive part of the loop
			//Parentheses are necessery for performance, otherwise EIGEN does the operation wrongly from left to right

			v = sigma_resid * h + (chol_ip_cross_cov.transpose() * (chol_ip_cross_cov * h));
			
			
			a = r.transpose() * z;
			a /= h.transpose() * v;

			u += a * h;
			r_old = r;
			r -= a * v;

			r_squared_norm = r.norm();
			//Log::REInfo("r.norm(): %g | Iteration: %i", r_squared_norm, j);
			if (std::isnan(r_squared_norm) || std::isinf(r_squared_norm)) {
				NaN_found = true;
				return;
			}
			if (r_squared_norm < delta_conv) {
				early_stop_alg = true;
			}

			z_old = z;

			//z = P^(-1) r 
			if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
				diag_sigma_resid_inv_r = diagonal_approx_inv_preconditioner.asDiagonal() * r; // ??? cwiseProd (TODO)
				sigma_cross_cov_diag_sigma_resid_inv_r = sigma_cross_cov.transpose() * diag_sigma_resid_inv_r;
				z = diag_sigma_resid_inv_r - (diagonal_approx_inv_preconditioner.asDiagonal() * (sigma_cross_cov * chol_fact_woodbury_preconditioner.solve(sigma_cross_cov_diag_sigma_resid_inv_r)));
				
			}
			else if (cg_preconditioner_type == "none") {
				z = r;
			}
			else {
				Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
			}
			
			b = r.transpose() * z;
			b /= r_old.transpose() * z_old;

			h = z + b * h;

			if (early_stop_alg) {

				Log::REInfo("CGFSA stop after %i CG-Iterations.", j+1);

				return;
			}
		}
		Log::REInfo("CGFSA used all %i iterations!", p);
		Log::REInfo("final r.norm(): %g", r_squared_norm);
	}// end CGFSA

	/*!
	* \brief Preconditioned conjugate gradient descent in combination with the Lanczos algorithm
	*		 Given the linear system AU=rhs where rhs is a matrix of dimension nxt of t probe column-vectors and
	*		 A=(C_s + C_nm*(C_m)^(-1)*C_mn) is a symmetric matrix of dimension nxn and a Full Scale Approximation for Sigma^-1
	*		 P = diag(C_s) + C_nm*(C_m)^(-1)*C_mn or P = mean(diag(C_s)) + C_nm*(C_m)^(-1)*C_mn is used as preconditioner.
	*		 The function returns t approximative tridiagonalizations T of the symmetric matrix A=QTQ' in vector form (diagonal + subdiagonal of T).
	* \param sigma_resid Residual Matrix C_s
	* \param sigma_cross_cov Matrix C_mn in Predictive Process Part C_nm*(C_m)^(-1)*C_mn
	* \param sigma_ip Matrix C_m in Predictive Process Part C_nm*(C_m)^(-1)*C_mn
	* \param chol_ip_cross_cov Cholesky Factor of C_m, the inducing point matrix, times cross-covariance
	* \param rhs Matrix of dimension nxt that contains (column-)probe vectors z_1,...,z_t with Cov[z_i] = P
	* \param[out] Tdiags The diagonals of the t approximative tridiagonalizations of A in vector form (solution written on input)
	* \param[out] Tsubdiags The subdiagonals of the t approximative tridiagonalizations of A in vector form (solution written on input)
	* \param[out] U Approximative solution of the linear system (solution written on input) (must have been declared with the correct nxt dimensions)
	* \param[out] NaN_found Is set to true, if NaN is found in the residual of conjugate gradient algorithm
	* \param num_data n-Dimension of the linear system
	* \param t t-Dimension of the linear system
	* \param p Number of conjugate gradient steps
	* \param delta_conv Tolerance for checking convergence of the algorithm
	* \param cg_preconditioner_type Type of preconditoner used for the conjugate gradient algorithm
	*/
	template <class T_mat>
	void CGTridiagFSA(const T_mat& sigma_resid,
		const den_mat_t& sigma_cross_cov,
		const den_mat_t& chol_ip_cross_cov,
		const den_mat_t& rhs,
		std::vector<vec_t>& Tdiags,
		std::vector<vec_t>& Tsubdiags,
		den_mat_t& U,
		bool& NaN_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const string_t cg_preconditioner_type,
		const chol_den_mat_t& chol_fact_woodbury_preconditioner,
		const vec_t& diagonal_approx_inv_preconditioner) {
		p = std::min(p, (int)num_data);

		den_mat_t R(num_data, t), R_old, Z(num_data, t), Z_old, H, V(num_data, t), diag_sigma_resid_inv_R, sigma_cross_cov_diag_sigma_resid_inv_R,
			mean_diag_sigma_resid_inv_R, sigma_cross_cov_mean_diag_sigma_resid_inv_R; //NEW V(num_data, t)
		vec_t v1(num_data);
		vec_t a(t), a_old(t);
		vec_t b(t), b_old(t);
		bool early_stop_alg = false;
		double mean_squared_R_norm;
		
		U.setZero();
		v1.setOnes();
		a.setOnes();
		b.setZero();

		bool is_zero = U.isZero(0);

		if (is_zero) {
			R = rhs;
		}
		else {
			R = rhs - (chol_ip_cross_cov.transpose() * (chol_ip_cross_cov * U));
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				R.col(i) -= sigma_resid * U.col(i); //parallelization in for loop is much faster
			}
		}
		//Z = P^(-1) R 
		if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
			//D^-1*R
			diag_sigma_resid_inv_R = diagonal_approx_inv_preconditioner.asDiagonal() * R;
			//Cmn*D^-1*R
			sigma_cross_cov_diag_sigma_resid_inv_R = sigma_cross_cov.transpose() * diag_sigma_resid_inv_R;
			//P^-1*R using Woodbury Identity
			Z = diag_sigma_resid_inv_R - (diagonal_approx_inv_preconditioner.asDiagonal() * (sigma_cross_cov * chol_fact_woodbury_preconditioner.solve(sigma_cross_cov_diag_sigma_resid_inv_R)));
			
		}
		else if (cg_preconditioner_type == "none") {
			Z = R;
		}
		else {
			Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
		}

		H = Z;
		for (int j = 0; j < p; ++j) {
			
			V = (chol_ip_cross_cov.transpose() * (chol_ip_cross_cov * H));
			
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				V.col(i) += sigma_resid * H.col(i); 
			}
			a_old = a;
			a = (R.cwiseProduct(Z).transpose() * v1).array() * (H.cwiseProduct(V).transpose() * v1).array().inverse(); //cheap

			U += H * a.asDiagonal();
			R_old = R;
			R -= V * a.asDiagonal();

			mean_squared_R_norm = 0;
//#pragma omp parallel for schedule(static)            
//			for (int i = 0; i < t; ++i) {
//				mean_squared_R_norm += R.col(i).norm();
//			}
//			mean_squared_R_norm /= t;
			mean_squared_R_norm = R.colwise().norm().mean();
			//Log::REInfo("mean_squared_R_norm: %g | Iteration: %i", mean_squared_R_norm, j);
			if (std::isnan(mean_squared_R_norm) || std::isinf(mean_squared_R_norm)) {
				NaN_found = true;
				return;
			}
			if (mean_squared_R_norm < delta_conv) {
				early_stop_alg = true;
			}
			
			Z_old = Z;

			if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
				diag_sigma_resid_inv_R = diagonal_approx_inv_preconditioner.asDiagonal() * R; 
				//Cmn*D^-1*R
				sigma_cross_cov_diag_sigma_resid_inv_R = sigma_cross_cov.transpose() * diag_sigma_resid_inv_R;
				//P^-1*R using Woodbury Identity
				Z = diag_sigma_resid_inv_R - (diagonal_approx_inv_preconditioner.asDiagonal() * (sigma_cross_cov * chol_fact_woodbury_preconditioner.solve(sigma_cross_cov_diag_sigma_resid_inv_R)));
				
			}
			else if (cg_preconditioner_type == "none") {
				Z = R;
			}
			else {
				Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
			}
			
			b_old = b;
			b = (R.cwiseProduct(Z).transpose() * v1).array() * (R_old.cwiseProduct(Z_old).transpose() * v1).array().inverse();

			H = Z + H * b.asDiagonal();
#pragma omp parallel for schedule(static)
			for (int i = 0; i < t; ++i) {
				Tdiags[i][j] = 1 / a(i) + b_old(i) / a_old(i);
				if (j > 0) {
					Tsubdiags[i][j - 1] = sqrt(b_old(i)) / a_old(i);
				}
			}
			if (early_stop_alg) {
				Log::REInfo("CGTridiagFSA stop after %i CG-Iterations.", j+1);
				Log::REInfo("final r.norm(): %g", mean_squared_R_norm);
				for (int i = 0; i < t; ++i) {
					Tdiags[i].conservativeResize(j + 1, 1);
					Tsubdiags[i].conservativeResize(j, 1);
				}
				return;
			}
		}
		Log::REInfo("CGTridiagFSA used all %i iterations!", p);
		Log::REInfo("final mean_squared_R_norm: %g", mean_squared_R_norm);
	} // end CGTridiagFSA


	template <class T_mat>
	void CGFSAMULTI(const T_mat& sigma_resid,
		const den_mat_t& sigma_cross_cov,
		const chol_den_mat_t& chol_fact_sigma_ip,
		const den_mat_t& rhs,
		den_mat_t& U,
		bool& NaN_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const string_t cg_preconditioner_type,
		const chol_den_mat_t& chol_fact_woodbury_preconditioner,
		const vec_t& diagonal_approx_inv_preconditioner) {
		p = std::min(p, (int)num_data);

		den_mat_t R(num_data, t), R_old, Z(num_data, t), Z_old, H, V(num_data, t), diag_sigma_resid_inv_R, sigma_cross_cov_diag_sigma_resid_inv_R,
			mean_diag_sigma_resid_inv_R, sigma_cross_cov_mean_diag_sigma_resid_inv_R; //NEW V(num_data, t)
		vec_t v1(num_data);
		vec_t a(t), a_old(t);
		vec_t b(t), b_old(t);
		bool early_stop_alg = false;
		double mean_squared_R_norm;
		std::chrono::milliseconds timespan(100); //temp

		U.setZero();
		v1.setOnes();
		a.setOnes();
		b.setZero();

		bool is_zero = U.isZero(0);

		if (is_zero) {
			R = rhs;
		}
		else {
			R = rhs - (sigma_cross_cov * (chol_fact_sigma_ip.solve(sigma_cross_cov.transpose() * U)));
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				R.col(i) -= sigma_resid * U.col(i); //parallelization in for loop is much faster
			}
		}
		//Z = P^(-1) R 
		if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
			//D^-1*R
			diag_sigma_resid_inv_R = diagonal_approx_inv_preconditioner.asDiagonal() * R;
			//Cmn*D^-1*R
			sigma_cross_cov_diag_sigma_resid_inv_R = sigma_cross_cov.transpose() * diag_sigma_resid_inv_R;
			//P^-1*R using Woodbury Identity
			Z = diag_sigma_resid_inv_R - (diagonal_approx_inv_preconditioner.asDiagonal() * (sigma_cross_cov * chol_fact_woodbury_preconditioner.solve(sigma_cross_cov_diag_sigma_resid_inv_R)));

		}
		else if (cg_preconditioner_type == "none") {
			Z = R;
		}
		else {
			Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
		}

		H = Z;

		for (int j = 0; j < p; ++j) {
			//The following matrix-vector operation is the expensive part of the loop
			//Parentheses are necessery for performance, otherwise EIGEN does the operation wrongly from left to right

			V = (sigma_cross_cov * (chol_fact_sigma_ip.solve(sigma_cross_cov.transpose() * H)));

#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				V.col(i) += sigma_resid * H.col(i);
			}
			a_old = a;
			a = (R.cwiseProduct(Z).transpose() * v1).array() * (H.cwiseProduct(V).transpose() * v1).array().inverse(); //cheap

			U += H * a.asDiagonal();
			R_old = R;
			R -= V * a.asDiagonal();

			mean_squared_R_norm = 0;
			//#pragma omp parallel for schedule(static)            
			//			for (int i = 0; i < t; ++i) {
			//				mean_squared_R_norm += R.col(i).norm();
			//			}
			//			mean_squared_R_norm /= t;
			mean_squared_R_norm = R.colwise().norm().mean();
			//Log::REInfo("mean_squared_R_norm: %g | Iteration: %i", mean_squared_R_norm, j);
			if (std::isnan(mean_squared_R_norm) || std::isinf(mean_squared_R_norm)) {
				NaN_found = true;
				return;
			}
			if (mean_squared_R_norm < delta_conv) {
				early_stop_alg = true;
			}

			Z_old = Z;

			if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
				diag_sigma_resid_inv_R = diagonal_approx_inv_preconditioner.asDiagonal() * R;
				//Cmn*D^-1*R
				sigma_cross_cov_diag_sigma_resid_inv_R = sigma_cross_cov.transpose() * diag_sigma_resid_inv_R;
				//P^-1*R using Woodbury Identity
				Z = diag_sigma_resid_inv_R - (diagonal_approx_inv_preconditioner.asDiagonal() * (sigma_cross_cov * chol_fact_woodbury_preconditioner.solve(sigma_cross_cov_diag_sigma_resid_inv_R)));

			}
			else if (cg_preconditioner_type == "none") {
				Z = R;
			}
			else {
				Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
			}

			b_old = b;
			b = (R.cwiseProduct(Z).transpose() * v1).array() * (R_old.cwiseProduct(Z_old).transpose() * v1).array().inverse();

			H = Z + H * b.asDiagonal();

			if (early_stop_alg) {

				Log::REInfo("CGFSAMULTI stop after %i CG-Iterations.", j + 1);

				return;
			}
		}
		Log::REInfo("CGFSAMULTI used all %i iterations!", p);
		Log::REInfo("final mean_squared_R_norm: %g", mean_squared_R_norm);
	} // end CGFSAMULTI

	template <class T_mat>
	void CGFSA_VARM(const T_mat& sigma_resid,
		const den_mat_t& rhs,
		den_mat_t& U,
		bool& NaN_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const string_t cg_preconditioner_type,
		const vec_t& diagonal_approx_inv_preconditioner) {
		p = std::min(p, (int)num_data);

		den_mat_t R(num_data, t), R_old, Z(num_data, t), Z_old, H, V(num_data, t), diag_sigma_resid_inv_R, sigma_cross_cov_diag_sigma_resid_inv_R,
			mean_diag_sigma_resid_inv_R, sigma_cross_cov_mean_diag_sigma_resid_inv_R; //NEW V(num_data, t)
		vec_t v1(num_data);
		vec_t a(t), a_old(t);
		vec_t b(t), b_old(t);
		bool early_stop_alg = false;
		double mean_squared_R_norm;
		std::chrono::milliseconds timespan(100); //temp

		U.setZero();
		v1.setOnes();
		a.setOnes();
		b.setZero();

		bool is_zero = U.isZero(0);

		if (is_zero) {
			R = rhs;
		}
		else {
			R = rhs;
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				R.col(i) -= sigma_resid * U.col(i); //parallelization in for loop is much faster
			}
		}
		//Z = P^(-1) R 
		if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
			//D^-1*R
			Z = diagonal_approx_inv_preconditioner.asDiagonal() * R;

		}
		else if (cg_preconditioner_type == "none") {
			Z = R;
		}
		else {
			Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
		}

		H = Z;

		for (int j = 0; j < p; ++j) {
			//The following matrix-vector operation is the expensive part of the loop
			//Parentheses are necessery for performance, otherwise EIGEN does the operation wrongly from left to right

			V.setZero();

#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				V.col(i) += sigma_resid * H.col(i);
			}
			a_old = a;
			a = (R.cwiseProduct(Z).transpose() * v1).array() * (H.cwiseProduct(V).transpose() * v1).array().inverse(); //cheap

			U += H * a.asDiagonal();
			R_old = R;
			R -= V * a.asDiagonal();

			mean_squared_R_norm = 0;
			//#pragma omp parallel for schedule(static)            
			//			for (int i = 0; i < t; ++i) {
			//				mean_squared_R_norm += R.col(i).norm();
			//			}
			//			mean_squared_R_norm /= t;
			mean_squared_R_norm = R.colwise().norm().mean();
			//Log::REInfo("mean_squared_R_norm: %g | Iteration: %i", mean_squared_R_norm, j);
			if (std::isnan(mean_squared_R_norm) || std::isinf(mean_squared_R_norm)) {
				NaN_found = true;
				return;
			}
			if (mean_squared_R_norm < delta_conv) {
				early_stop_alg = true;
			}

			Z_old = Z;

			if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
				Z = diagonal_approx_inv_preconditioner.asDiagonal() * R;

			}
			else if (cg_preconditioner_type == "none") {
				Z = R;
			}
			else {
				Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
			}

			b_old = b;
			b = (R.cwiseProduct(Z).transpose() * v1).array() * (R_old.cwiseProduct(Z_old).transpose() * v1).array().inverse();

			H = Z + H * b.asDiagonal();

			if (early_stop_alg) {

				Log::REInfo("CGFSA_VARM stop after %i CG-Iterations.", j + 1);

				return;
			}
		}
		Log::REInfo("CGFSA_VARM used all %i iterations!", p);
		Log::REInfo("final mean_squared_R_norm: %g", mean_squared_R_norm);
	} // end CGFSAMULTI


	/*!
	* \brief Lanczos algorithm with full reorthogonalization to approximately factorize the symmetric matrix (C_l + C_s) as Q_k T_k Q_k'
	*		 where T_k is a tridiagonal matrix of dimension kxk and Q_k a matrix of dimension nxk. The diagonal and subdiagonal of T_k is returned in vector form.
	* \param sigma_resid Residual Matrix C_s
	* \param sigma_cross_cov Matrix C_mn in Predictive Process Part C_nm*(C_m)^(-1)*C_mn
	* \param chol_fact_sigma_ip Cholesky Factor of C_m, the inducing point matrix
	* \param b_init Inital column-vector of Q_k (after normalization) of dimension nx1.
	* \param num_data n-Dimension
	* \param[out] Tdiag_k The diagonal of the tridiagonal matrix T_k (solution written on input) (must have been declared with the correct kx1 dimension)
	* \param[out] Tsubdiag_k The subdiagonal of the tridiagonal matrix T_k (solution written on input) (must have been declared with the correct (k-1)x1 dimension)
	* \param[out] Q_k Matrix Q_k = [b_init/||b_init||, q_2, q_3, ...] (solution written on input) (must have been declared with the correct nxk dimensions)
	* \param max_it Rank k of the matrix Q_k and T_k
	* \param tol Tolerance to decide wether reorthogonalization is necessary
	*/
	template <class T_mat>
	void LanczosTridiagFSA(const T_mat& sigma_resid,
		const den_mat_t& sigma_cross_cov,
		const chol_den_mat_t& chol_fact_sigma_ip,
		const vec_t& b_init,
		const data_size_t num_data,
		vec_t& Tdiag_k,
		vec_t& Tsubdiag_k,
		den_mat_t& Q_k,
		int max_it,
		const double tol) {

		bool could_reorthogonalize;
		int final_rank = 1;
		double alpha_0, beta_0, alpha_curr, beta_curr, beta_prev;
		vec_t q_0, q_curr, q_prev, r, inner_products;
		
		max_it = std::min(max_it, num_data);

		//Preconditioning
		vec_t diag_sigma_resid_inv_q, sigma_cross_cov_diag_sigma_resid_inv_q,
			mean_diag_sigma_resid_inv_q, sigma_cross_cov_mean_diag_sigma_resid_inv_q;

		//Inital vector of Q_k: q_0
		Q_k.resize(num_data, max_it);
		q_0 = b_init / b_init.norm();
		Q_k.col(0) = q_0;

		//(C_s + CnmCm-1Cmn) q_0
		r = sigma_resid * q_0 + (sigma_cross_cov * (chol_fact_sigma_ip.solve(sigma_cross_cov.transpose() * q_0)));
		
		//Initial alpha value: alpha_0
		alpha_0 = q_0.dot(r);
		
		//Initial beta value: beta_0
		r -= alpha_0 * q_0;
		beta_0 = r.norm();
		
		//Store alpha_0 and beta_0 into T_k
		Tdiag_k(0) = alpha_0;
		Tsubdiag_k(0) = beta_0;
		
		//Compute next vector of Q_k: q_1
		Q_k.col(1) = r / beta_0;
		
		//Start the iterations
		for (int k = 1; k < max_it; ++k) {
			//Log::REInfo("k: %i", k);
			//Get previous values
			q_prev = Q_k.col(k - 1);
			q_curr = Q_k.col(k);
			beta_prev = Tsubdiag_k(k - 1);

			//Compute next alpha value
			r = sigma_resid * q_curr + (sigma_cross_cov * (chol_fact_sigma_ip.solve(sigma_cross_cov.transpose() * q_curr))) - beta_prev * q_prev;
			
			alpha_curr = q_curr.dot(r);
			
			//Store alpha_curr
			Tdiag_k(k) = alpha_curr;
			final_rank += 1;

			if ((k + 1) < max_it) {

				//Compute next residual
				r -= alpha_curr * q_curr;
				
				//Full reorthogonalization: r = r - Q_k*(Q_k' r)
				r -= Q_k(Eigen::all, Eigen::seq(0, k)) * (Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r);

				//Compute next beta value
				beta_curr = r.norm();
				Tsubdiag_k(k) = beta_curr;

				r /= beta_curr;

				//More reorthogonalizations if necessary
				inner_products = Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r;
				could_reorthogonalize = false;
				for (int l = 0; l < 10; ++l) {
					if ((inner_products.array() < tol).all()) {
						could_reorthogonalize = true;
						break;
					}
					Log::REInfo("Rereorthogonalize");
					r -= Q_k(Eigen::all, Eigen::seq(0, k)) * (Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r);
					r /= r.norm();
					inner_products = Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r;
				}

				//Store next vector of Q_k
				Q_k.col(k + 1) = r;

				if (abs(beta_curr) < 1e-6 || !could_reorthogonalize) {
					break;
				}
			}
		}

		//Resize Q_k, Tdiag_k, Tsubdiag_k
		Log::REInfo("final rank: %i", final_rank);
		Q_k.conservativeResize(num_data, final_rank);
		Tdiag_k.conservativeResize(final_rank, 1);
		Tsubdiag_k.conservativeResize(final_rank - 1, 1);

	}// end LanczosTridiagFSA

	/*!
	* \brief Lanczos algorithm with full reorthogonalization to approximately factorize the symmetric matrix (C_l + C_s) as Q_k T_k Q_k'
	*		 where T_k is a tridiagonal matrix of dimension kxk and Q_k a matrix of dimension nxk. The diagonal and subdiagonal of T_k is returned in vector form.
	* \param sigma_resid Residual Matrix C_s
	* \param b_init Inital column-vector of Q_k (after normalization) of dimension nx1.
	* \param num_data n-Dimension
	* \param[out] Tdiag_k The diagonal of the tridiagonal matrix T_k (solution written on input) (must have been declared with the correct kx1 dimension)
	* \param[out] Tsubdiag_k The subdiagonal of the tridiagonal matrix T_k (solution written on input) (must have been declared with the correct (k-1)x1 dimension)
	* \param[out] Q_k Matrix Q_k = [b_init/||b_init||, q_2, q_3, ...] (solution written on input) (must have been declared with the correct nxk dimensions)
	* \param max_it Rank k of the matrix Q_k and T_k
	* \param tol Tolerance to decide wether reorthogonalization is necessary
	*/
	template <class T_mat>
	void LanczosTridiagResid(const T_mat& sigma_resid,
		const vec_t& b_init,
		const data_size_t num_data,
		vec_t& Tdiag_k,
		vec_t& Tsubdiag_k,
		den_mat_t& Q_k,
		int max_it,
		const double tol) {
		bool could_reorthogonalize;
		int final_rank = 1;
		double alpha_0, beta_0, alpha_curr, beta_curr, beta_prev;
		vec_t q_0, q_curr, q_prev, r, inner_products;
		vec_t diag_resid = sigma_resid.diagonal().cwiseSqrt().cwiseInverse();
		max_it = std::min(max_it, num_data);

		//Preconditioning
		vec_t diag_sigma_resid_inv_q, sigma_cross_cov_diag_sigma_resid_inv_q,
			mean_diag_sigma_resid_inv_q, sigma_cross_cov_mean_diag_sigma_resid_inv_q;

		//Inital vector of Q_k: q_0
		Q_k.resize(num_data, max_it);
		q_0 = b_init / b_init.norm();
		Q_k.col(0) = q_0;
		//C_s q_0
		r = sigma_resid* q_0;
		
		//Initial alpha value: alpha_0
		alpha_0 = q_0.dot(r);

		//Initial beta value: beta_0
		r -= alpha_0 * q_0;
		beta_0 = r.norm();
		//Store alpha_0 and beta_0 into T_k
		Tdiag_k(0) = alpha_0;
		Tsubdiag_k(0) = beta_0;

		//Compute next vector of Q_k: q_1
		Q_k.col(1) = r / beta_0;
		//Start the iterations
		for (int k = 1; k < max_it; ++k) {
			//Log::REInfo("k: %i", k);
			//Get previous values
			q_prev = Q_k.col(k - 1);
			q_curr = Q_k.col(k);
			beta_prev = Tsubdiag_k(k - 1);

			//Compute next alpha value
			r = sigma_resid * q_curr - beta_prev * q_prev;
			
			alpha_curr = q_curr.dot(r);

			//Store alpha_curr
			Tdiag_k(k) = alpha_curr;
			final_rank += 1;

			if ((k + 1) < max_it) {

				//Compute next residual
				r -= alpha_curr * q_curr;

				//Full reorthogonalization: r = r - Q_k*(Q_k' r)
				r -= Q_k(Eigen::all, Eigen::seq(0, k)) * (Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r);

				//Compute next beta value
				beta_curr = r.norm();
				Tsubdiag_k(k) = beta_curr;

				r /= beta_curr;

				//More reorthogonalizations if necessary
				inner_products = Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r;
				could_reorthogonalize = false;
				for (int l = 0; l < 10; ++l) {
					if ((inner_products.array() < tol).all()) {
						could_reorthogonalize = true;
						break;
					}
					Log::REInfo("Rereorthogonalize");
					r -= Q_k(Eigen::all, Eigen::seq(0, k)) * (Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r);
					r /= r.norm();
					inner_products = Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r;
				}

				//Store next vector of Q_k
				Q_k.col(k + 1) = r;

				if (abs(beta_curr) < 1e-6 || !could_reorthogonalize) {
					break;
				}
			}
		}

		//Resize Q_k, Tdiag_k, Tsubdiag_k
		Log::REInfo("final rank: %i", final_rank);
		Q_k.conservativeResize(num_data, final_rank);
		Tdiag_k.conservativeResize(final_rank, 1);
		Tsubdiag_k.conservativeResize(final_rank - 1, 1);
	}// end LanczosTridiagResid

	/*!
	* \brief Fills a matrix with Rademacher -1/+1 random numbers or normal samples
	* \param generator Random number generator
	* \param[out] Z Rademacher/Normal matrix. Z need to be defined before calling this function
	* \param rademacher if true Rademacher samples else normal samples
	*/
	void simProbeVect(RNG_t& generator,
		den_mat_t& Z, 
		const bool rademacher);
	
	/*!
	* \brief Stochastic estimation of log(det(A)) given t approximative tridiagonalizations T of a symmetric matrix A=QTQ' of dimension nxn,
	*		 where T is given in vector form (diagonal + subdiagonal of T).
	* \param Tdiags The diagonals of the t approximative tridiagonalizations of A in vector form
	* \param Tsubdiags The subdiagonals of the t approximative tridiagonalizations of A in vector form
	* \param ldet[out] Estimation of log(det(A)) (solution written on input)
	* \param num_data Number of data points
	* \param t Number of tridiagonalization matrices T
	*/
	void LogDetStochTridiag(const std::vector<vec_t>& Tdiags,
		const  std::vector<vec_t>& Tsubdiags,
		double& ldet,
		const data_size_t num_data,
		const int t);

	/*!
	* \brief Stochastic trace estimation for tr(A^-1 * A_Gradient) with variance reduction with Full Scale Approximation A = C_s + C_nm * (C_m)^-1 * C_mn
	* \param A_G_Z Matrix A_Gradient of dimension nxn multiplied with the matrix Z = (z_1,...,z_t) of dimension nxt that contains the random probe vectors
	* \param A_inv_Z Solution of AX = Z
	* \param P_G_Z Derivative of Preconditioner P_Gradient of dimension nxn multiplied with the matrix Z = (z_1,...,z_t) of dimension nxt that contains the random probe vectors
	* \param P_inv_Z Solution of PX = Z with Preconditioner P
	* \param num_data Number of data points
	* \param tr[out] Vector of stochastic trace tr(A^-1 * A_Gradient) (solution written on input)
	* \param sigma_ip_grad Gradient of inducing point matrix C_m
	* \param sigma_cross_cov_grad Gradient of cross covariance matrix C_mn
	* \param sigma_cross_cov Cross covariance matrix C_mn
	* \param sigma_resid_grad Gradient of residual covariance matrix C_s
	* \param chol_fact_woodbury_sigma Cholesky Factor of Woodbury Matrix for Inverse of Preconditioner
	* \param chol_fact_sigma_ip Cholesky Factor of inducing point matrix C_m
	* \param diag_sigma_res Diagonal of residual covariance C_s
	* \param control_variates If true then apply control variates
	* \param cg_preconditioner_type Type of preconditoner used for the conjugate gradient algorithm
	* \param variance_reduction If true then apply var reductio 
	*/
	template <class T_mat>
	void StochTraceFSA(const den_mat_t& A_G_Z,
		const den_mat_t& A_inv_Z,
		const den_mat_t& P_G_Z,
		const den_mat_t& P_inv_Z,
		const data_size_t num_data,
		double& tr,
		const den_mat_t& sigma_ip_inv_sigma_ip_stable_grad,
		const T_mat& sigma_resid_grad,
		const chol_den_mat_t& chol_fact_woodbury_preconditioner,
		const den_mat_t& sigma_woodbury_preconditioner_grad,
		const vec_t& diagonal_approx_inv_preconditioner,
		const bool control_variates,
		const string_t cg_preconditioner_type,
		const bool variance_reduction) {

		// Trace without Preconditioner part or
		// First term : Stochastic Trace of Sigma ^ -1 * Sigma_Grad with Preconditioner sample vectors
		vec_t Schur_Prod_A = (A_inv_Z.cwiseProduct(A_G_Z)).colwise().sum();
		tr = Schur_Prod_A.mean();

		if (cg_preconditioner_type != "none" && variance_reduction) {
			// Initialization
			vec_t a, b, b1, diagonal_approx_grad_preconditioner;
			double Tr_P = 0;
			double mean_diagonal_approx_grad_preconditioner, Tr_P_stoch, c;

			// Second term: Exact Trace of P^-1*P_Grad with Preconditioner P
			Tr_P -= sigma_ip_inv_sigma_ip_stable_grad.trace();
			if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
				diagonal_approx_grad_preconditioner = sigma_resid_grad.diagonal();
				Tr_P += diagonal_approx_inv_preconditioner.dot(diagonal_approx_grad_preconditioner);
			}
			den_mat_t sigma_woodbury_inv_sigma_woodbury_stable_grad = chol_fact_woodbury_preconditioner.solve(sigma_woodbury_preconditioner_grad);
			Tr_P += sigma_woodbury_inv_sigma_woodbury_stable_grad.trace();

			// Third term: Stochastic Trace of P^-1*P_Grad with Preconditioner P
			vec_t Schur_Prod_P = (P_inv_Z.cwiseProduct(P_G_Z)).colwise().sum();
			Tr_P_stoch = Schur_Prod_P.mean();


			// Control Variates
			if (control_variates) {
				b = Schur_Prod_P.array() - Tr_P;
				a = Schur_Prod_A.array() - Schur_Prod_A.mean();
				c = (a.dot(b)) / (b.dot(b));
				Log::REInfo("The optimal c is %g", c);
				tr += c * (Tr_P - Tr_P_stoch);
			}
			else {
				tr += Tr_P - Tr_P_stoch;
			}
		}
	}//end StochTraceVarRedFSA
}
#endif   // GPB_CG_UTILS_
