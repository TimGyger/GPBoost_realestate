% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/GPModel.R
\name{fitGPModel}
\alias{fitGPModel}
\title{Fits a \code{GPModel}}
\usage{
fitGPModel(likelihood = "gaussian", group_data = NULL,
  group_rand_coef_data = NULL, ind_effect_group_rand_coef = NULL,
  drop_intercept_group_rand_effect = NULL, gp_coords = NULL,
  gp_rand_coef_data = NULL, cov_function = "exponential",
  cov_fct_shape = 0.5, gp_approx = "none", cov_fct_taper_range = 1,
  cov_fct_taper_shape = 0, num_neighbors = 30L,
  vecchia_ordering = "random", vecchia_pred_type = NULL,
  num_neighbors_pred = num_neighbors, num_ind_points = 500L,
  matrix_inversion_method = "cholesky", seed = 0L, cluster_ids = NULL,
  free_raw_data = FALSE, y, X = NULL, params = list(),
  vecchia_approx = NULL)
}
\arguments{
\item{likelihood}{A \code{string} specifying the likelihood function (distribution) of the response variable
Default = "gaussian"}

\item{group_data}{A \code{vector} or \code{matrix} whose columns are categorical grouping variables. 
The elements being group levels defining grouped random effects.
The elements of 'group_data' can be integer, double, or character.
The number of columns corresponds to the number of grouped (intercept) random effects}

\item{group_rand_coef_data}{A \code{vector} or \code{matrix} with numeric covariate data 
for grouped random coefficients}

\item{ind_effect_group_rand_coef}{A \code{vector} with integer indices that 
indicate the corresponding categorical grouping variable (=columns) in 'group_data' for 
every covariate in 'group_rand_coef_data'. Counting starts at 1.
The length of this index vector must equal the number of covariates in 'group_rand_coef_data'.
For instance, c(1,1,2) means that the first two covariates (=first two columns) in 'group_rand_coef_data'
have random coefficients corresponding to the first categorical grouping variable (=first column) in 'group_data',
and the third covariate (=third column) in 'group_rand_coef_data' has a random coefficient
corresponding to the second grouping variable (=second column) in 'group_data'}

\item{drop_intercept_group_rand_effect}{A \code{vector} of type \code{logical} (boolean). 
Indicates whether intercept random effects are dropped (only for random coefficients). 
If drop_intercept_group_rand_effect[k] is TRUE, the intercept random effect number k is dropped / not included. 
Only random effects with random slopes can be dropped.}

\item{gp_coords}{A \code{matrix} with numeric coordinates (= inputs / features) for defining Gaussian processes}

\item{gp_rand_coef_data}{A \code{vector} or \code{matrix} with numeric covariate data for
Gaussian process random coefficients}

\item{cov_function}{A \code{string} specifying the covariance function for the Gaussian process. 
Available options:
"exponential", "gaussian", "matern", "powered_exponential", "wendland"
For "exponential", "gaussian", and "powered_exponential", we follow the notation and parametrization of Diggle and Ribeiro (2007).
For "matern", we follow the notation of Rassmusen and Williams (2006).
For "wendland", we follow the notation of Bevilacqua et al. (2019, AOS)}

\item{cov_fct_shape}{A \code{numeric} specifying the shape parameter of the covariance function 
(=smoothness parameter for Matern covariance)  
This parameter is irrelevant for some covariance functions such as the exponential or Gaussian}

\item{gp_approx}{A \code{string} specifying the large data approximation
for Gaussian processes. Available options: 
\itemize{
\item{"none"}{ No approximation }
\item{"vecchia"}{ A Vecchia approximation; see Sigrist (2022, JMLR for more details) }
\item{"tapering"}{ The covariance function is multiplied by 
a compactly supported Wendland correlation function }
}}

\item{cov_fct_taper_range}{A \code{numeric} specifying the range parameter 
of the Wendland covariance function and Wendland correlation taper function. 
We follow the notation of Bevilacqua et al. (2019, AOS)}

\item{cov_fct_taper_shape}{A \code{numeric} specifying the shape (=smoothness) parameter 
of the Wendland covariance function and Wendland correlation taper function. 
We follow the notation of Bevilacqua et al. (2019, AOS)}

\item{num_neighbors}{An \code{integer} specifying the number of neighbors for the Vecchia approximation}

\item{vecchia_ordering}{A \code{string} specifying the ordering used in the Vecchia approximation. 
Available options:
#' \itemize{
\item{"none"} { the default ordering in the data is used }
\item{"random"} { a random ordering }
}}

\item{vecchia_pred_type}{A \code{string} specifying the type of Vecchia approximation used for making predictions.
Default value if vecchia_pred_type = NULL: "order_obs_first_cond_obs_only"
Available options:
\itemize{
\item{"order_obs_first_cond_obs_only"}{ Vecchia approximation for the observable process and observed training data is 
ordered first and the neighbors are only observed training data points }
\item{"order_obs_first_cond_all"}{ Vecchia approximation for the observable process and observed training data is 
ordered first and the neighbors are selected among all points (training + prediction) }
\item{"latent_order_obs_first_cond_obs_only"}{ Vecchia approximation for the latent process and observed data is 
ordered first and neighbors are only observed points}
\item{"latent_order_obs_first_cond_all"}{ Vecchia approximation 
for the latent process and observed data is ordered first and neighbors are selected among all points }
\item{"order_pred_first"}{ Vecchia approximation for the observable process and prediction data is 
ordered first for making predictions. This option is only available for Gaussian likelihoods }
}}

\item{num_neighbors_pred}{an \code{integer} specifying the number of neighbors for the Vecchia approximation 
for making predictions. Default value if NULL: num_neighbors_pred = num_neighbors}

\item{num_ind_points}{An \code{integer} specifying the number of inducing 
points / knots for, e.g., a predictive process approximation}

\item{matrix_inversion_method}{A \code{string} specifying the method used for inverting covariance matrices
Available options:
\itemize{
\item{"cholesky"}{ Cholesky factorization }
}}

\item{seed}{An \code{integer} specifying the seed used for model creation 
(e.g., random ordering in Vecchia approximation)}

\item{cluster_ids}{A \code{vector} with elements indicating independent realizations of 
random effects / Gaussian processes (same values = same process realization).
The elements of 'cluster_ids' can be integer, double, or character.}

\item{free_raw_data}{A \code{boolean}. If TRUE, the data (groups, coordinates, covariate data for random coefficients) 
is freed in R after initialization}

\item{y}{A \code{vector} with response variable data}

\item{X}{A \code{matrix} with numeric covariate data for the 
fixed effects linear regression term (if there is one)}

\item{params}{A \code{list} with parameters for the estimation / optimization
 \itemize{
    \item{optimizer_cov}{ Optimizer used for estimating covariance parameters. 
    Options: "gradient_descent", "fisher_scoring", "nelder_mead", "bfgs", "adam".
    Default="gradient_descent"}
    \item{optimizer_coef}{ Optimizer used for estimating linear regression coefficients, if there are any 
    (for the GPBoost algorithm there are usually none). 
    Options: "gradient_descent", "wls", "nelder_mead", "bfgs", "adam". Gradient descent steps are done simultaneously 
    with gradient descent steps for the covariance parameters. 
    "wls" refers to doing coordinate descent for the regression coefficients using weighted least squares.
    Default="wls" for Gaussian data and "gradient_descent" for other likelihoods.
    If 'optimizer_cov' is set to "nelder_mead", "bfgs", or "adam", 'optimizer_coef' is automatically also set to the same value.}
    \item{maxit}{ Maximal number of iterations for optimization algorithm. Default=1000}
    \item{delta_rel_conv}{ Convergence tolerance. The algorithm stops if the relative change 
    in eiher the (approximate) log-likelihood or the parameters is below this value. 
    For "bfgs" and "adam", the L2 norm of the gradient is used instead of the relative change in the log-likelihood. 
    Default=1E-6}
    \item{convergence_criterion}{ The convergence criterion used for terminating the optimization algorithm.
    Options: "relative_change_in_log_likelihood" (default) or "relative_change_in_parameters"}
    \item{init_coef}{ Initial values for the regression coefficients (if there are any, can be NULL).
    Default=NULL}
    \item{init_cov_pars}{ Initial values for covariance parameters of Gaussian process and 
    random effects (can be NULL). Default=NULL}
    \item{lr_coef}{ Learning rate for fixed effect regression coefficients if gradient descent is used.
    Default=0.1}
    \item{lr_cov}{ Learning rate for covariance parameters. If <= 0, internal default values are used.
    Default value = 0.1 for "gradient_descent" and 1. for "fisher_scoring"}
    \item{use_nesterov_acc}{ If TRUE Nesterov acceleration is used.
    This is used only for gradient descent. Default=TRUE}
    \item{acc_rate_coef}{ Acceleration rate for regression coefficients (if there are any) 
    for Nesterov acceleration. Default=0.5}
    \item{acc_rate_cov}{ Acceleration rate for covariance parameters for Nesterov acceleration.
    Default=0.5}
    \item{momentum_offset}{ Number of iterations for which no momentum is applied in the beginning.
    Default=2}
    \item{trace}{ If TRUE, information on the progress of the parameter
    optimization is printed. Default=FALSE}
    \item{std_dev}{ If TRUE, approximate standard deviations are calculated for the covariance and linear regression parameters 
    (= square root of diagonal of the inverse Fisher information for Gaussian likelihoods and 
    square root of diagonal of a numerically approximated inverse Hessian for non-Gaussian likelihoods)}
}}

\item{vecchia_approx}{This is discontinued. Use the argument \code{gp_approx} instead}
}
\value{
A fitted \code{GPModel}
}
\description{
Estimates the parameters of a \code{GPModel} using maximum likelihood estimation
}
\examples{
# See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples

data(GPBoost_data, package = "gpboost")
# Add intercept column
X1 <- cbind(rep(1,dim(X)[1]),X)
X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)

#--------------------Grouped random effects model: single-level random effect----------------
gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1,
                       likelihood="gaussian", params = list(std_dev = TRUE))
summary(gp_model)
# Make predictions
pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
                X_pred = X_test1, predict_var = TRUE)
pred$mu # Predicted mean
pred$var # Predicted variances
# Also predict covariance matrix
pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
                X_pred = X_test1, predict_cov_mat = TRUE)
pred$mu # Predicted mean
pred$cov # Predicted covariance


\donttest{
#--------------------Two crossed random effects and a random slope----------------
gp_model <- fitGPModel(group_data = group_data, likelihood="gaussian",
                       group_rand_coef_data = X[,2],
                       ind_effect_group_rand_coef = 1,
                       y = y, X = X1, params = list(std_dev = TRUE))
summary(gp_model)


#--------------------Gaussian process model----------------
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                       likelihood="gaussian", y = y, X = X1, params = list(std_dev = TRUE))
summary(gp_model)
# Make predictions
pred <- predict(gp_model, gp_coords_pred = coords_test, 
                X_pred = X_test1, predict_cov_mat = TRUE)
pred$mu # Predicted (posterior) mean of GP
pred$cov # Predicted (posterior) covariance matrix of GP


#--------------------Gaussian process model with Vecchia approximation----------------
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                       gp_approx = "vecchia", num_neighbors = 30,
                       likelihood="gaussian", y = y)
summary(gp_model)


#--------------------Gaussian process model with random coefficients----------------
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                       gp_rand_coef_data = X[,2], y=y,
                       likelihood = "gaussian", params = list(std_dev = TRUE))
summary(gp_model)


#--------------------Combine Gaussian process with grouped random effects----------------
gp_model <- fitGPModel(group_data = group_data,
                       gp_coords = coords, cov_function = "exponential",
                       likelihood = "gaussian", y = y, X = X1, params = list(std_dev = TRUE))
summary(gp_model)
}

}
\author{
Fabio Sigrist
}
