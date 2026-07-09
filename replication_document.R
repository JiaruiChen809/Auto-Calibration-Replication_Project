library(mgcv)
library(gbm)
library(dplyr)
library(locfit)

getwd()

#download data
motor_data <- read.csv("MTPL_data.csv", sep = ";", header = TRUE)
str(motor_data)
head(motor_data)

motor_data$gas   <- as.factor(motor_data$gas)
motor_data$brand <- as.factor(motor_data$brand)
motor_data$area  <- as.factor(motor_data$area)
motor_data$ct    <- as.factor(motor_data$ct)

#split data
set.seed(123)

n <- nrow(motor_data)
idx_train <- sample(1:n, size = 0.8 * n)

D <- motor_data[idx_train, ]
D_hat <- motor_data[-idx_train, ]


#replication for 5.1

# Observed number of claims for each policyholder in the validation set
Y <- D_hat$claims
# True expected number of claims: m_i = e_i * mu(X_i)
m <- D_hat$expo * D_hat$truefreq

# Sort by fitted value m, implementing I[m_i <= F_m^{-1}(alpha)]
ord <- order(m)

Y_ord <- Y[ord]
m_ord <- m[ord]

n <- length(Y_ord)

# Empirical Concentration Curve
# CC_hat(alpha) = sum Y_i I[m_i <= q_alpha] / sum Y_i
CC <- cumsum(Y_ord) / sum(Y_ord)

# Empirical Lorenz Curve
# LC_hat(alpha) = sum m_i I[m_i <= q_alpha] / sum m_i
LC <- cumsum(m_ord) / sum(m_ord)

# x axis denoted to be alpha
alpha <- seq_len(n) / n

# plot two curves
plot(
  alpha, CC,
  type = "l",
  col = "red",
  lwd = 2,
  xlab = expression(alpha),
  ylab = "Curve",
  main = "True Model: Concentration Curve and Lorenz Curve"
)

lines(
  alpha, LC,
  col = "blue",
  lwd = 2
)

legend(
  "topleft",
  legend = c("Concentration Curve", "Lorenz Curve"),
  col = c("red", "blue"),
  lwd = 2,
  bty = "n"
)


#find p value for LC vs CC
p_equal_test <- function(Y, m, B = 500, seed = 123) {
  
  set.seed(seed)
  
  ord <- order(m)
  Y <- Y[ord]
  m <- m[ord]
  
  n <- length(Y)
  
  Ybar <- mean(Y)
  mbar <- mean(m)
  
  # a_i = Y_i / Ybar - m_i / mbar
  a <- Y / Ybar - m / mbar
  
  # T_n(alpha)
  Tn <- cumsum(a) / sqrt(n)
  T_obs <- max(abs(Tn))
  
  exceed <- numeric(B)
  
  for (b in 1:B) {
    U <- rnorm(n)
    
    Delta <- cumsum(a * U) / sqrt(n) -
      Tn * sum(U) / n
    
    exceed[b] <- max(abs(Delta)) > T_obs
  }
  
  mean(exceed)
}

p_equal <- p_equal_test(Y, m, B = 500)





#find p value for auto-calibration
p_auto_test <- function(Y, m, B = 500, seed = 123) {
  
  set.seed(seed)
  
  ord <- order(m)
  Y <- Y[ord]
  m <- m[ord]
  
  n <- length(Y)
  
  # b_i = Y_i - m_i
  b_i <- Y - m
  
  # A_n(alpha)
  An <- cumsum(b_i) / sqrt(n)
  A_obs <- max(abs(An))
  
  exceed <- numeric(B)
  
  for (b in 1:B) {
    U <- rnorm(n)
    
    Delta <- cumsum(b_i * U) / sqrt(n) -
      An * sum(U) / n
    
    exceed[b] <- max(abs(Delta)) > A_obs
  }
  
  mean(exceed)
}

p_auto <- p_auto_test(Y, m, B = 500)


p_equal
p_auto



#replication for models

#seperate D into D1 and D2,
#D1 to train the model and D2 to implement the auto-calibration procedure.
idx_D1 <- sample(1:nrow(D), size = 0.8 * nrow(D))
D1 <- D[idx_D1, ]
D2 <- D[-idx_D1, ]

# GAM1: only age
gam1 <- gam(
  claims ~ s(age) + offset(log(expo)),
  family = poisson(link = "log"),
  data = D1
)

# GAM2: all 8 features
gam2 <- gam(
  claims ~ s(age) + s(ac) + s(power) + gas + brand + area + s(dens) + ct +
    offset(log(expo)),
  family = poisson(link = "log"),
  data = D1
)

# Poisson GLM: all 8 features


glm1 <- glm(
  claims ~ age + ac + power + gas + brand + area + dens + ct +
    offset(log(expo)),
  family = poisson(link = "log"),
  data = D1
)

# Create noisy covariates for noisy GAMs

set.seed(123)

noise_level <- 0.10   # 10% of each variable's empirical SD

D1_noisy <- D1
D_hat_noisy <- D_hat

add_noise <- function(x, level = 0.10) {
  x + rnorm(length(x), mean = 0, sd = level * sd(x, na.rm = TRUE))
}

D1_noisy$age   <- add_noise(D1$age, noise_level)
D1_noisy$ac    <- add_noise(D1$ac, noise_level)
D1_noisy$power <- add_noise(D1$power, noise_level)
D1_noisy$dens  <- add_noise(D1$dens, noise_level)

D_hat_noisy$age   <- add_noise(D_hat$age, noise_level)
D_hat_noisy$ac    <- add_noise(D_hat$ac, noise_level)
D_hat_noisy$power <- add_noise(D_hat$power, noise_level)
D_hat_noisy$dens  <- add_noise(D_hat$dens, noise_level)


# GAM1 with noisy covariates
gam1_noise <- gam(
  claims ~ s(age) + offset(log(expo)),
  family = poisson(link = "log"),
  data = D1_noisy
)

# GAM2 with noisy covariates
gam2_noise <- gam(
  claims ~ s(age) + s(ac) + s(power) + gas + brand + area + s(dens) + ct +
    offset(log(expo)),
  family = poisson(link = "log"),
  data = D1_noisy
)








# Predicted claims on validation set

# GAM1 prediction
D_hat$pred_gam1 <- predict(gam1, newdata = D_hat, type = "response")
D_hat$freq_gam1 <- D_hat$pred_gam1 / D_hat$expo

p_equal_test(D_hat$claims, D_hat$pred_gam1)
p_auto_test(D_hat$claims, D_hat$pred_gam1)


# GAM2 prediction
D_hat$pred_gam2 <- predict(gam2, newdata = D_hat, type = "response")
D_hat$freq_gam2 <- D_hat$pred_gam2 / D_hat$expo

p_equal_test(D_hat$claims, D_hat$pred_gam2)
p_auto_test(D_hat$claims, D_hat$pred_gam2)


# GLM prediction
D_hat$pred_glm1 <- predict(glm1, newdata = D_hat, type = "response")

p_equal_test(D_hat$claims, D_hat$pred_glm1)
p_auto_test(D_hat$claims, D_hat$pred_glm1)


# GAM1_noise prediction
D_hat$pred_gam1_noise <- predict(gam1_noise, newdata = D_hat, type = "response")
D_hat$freq_gam1_noise <- D_hat$pred_gam1_noise / D_hat$expo

p_equal_test(D_hat$claims, D_hat$pred_gam1_noise)
p_auto_test(D_hat$claims, D_hat$pred_gam1_noise)


# GAM2_noise prediction
D_hat$pred_gam2_noise <- predict(gam2_noise, newdata = D_hat, type = "response")
D_hat$freq_gam2_noise <- D_hat$pred_gam2_noise / D_hat$expo

p_equal_test(D_hat$claims, D_hat$pred_gam2_noise)
p_auto_test(D_hat$claims, D_hat$pred_gam2_noise)


#build function for CC and LC curve
plot_CC_LC <- function(Y, pred,
                       main = "CC vs LC") {
  
  # Sort by predicted value
  ord <- order(pred)
  
  Y <- Y[ord]
  pred <- pred[ord]
  
  n <- length(Y)
  
  alpha <- seq_len(n) / n
  
  # Concentration Curve
  CC <- cumsum(Y) / sum(Y)
  
  # Lorenz Curve
  LC <- cumsum(pred) / sum(pred)
  
  plot(
    alpha, CC,
    type = "l",
    col = "red",
    lwd = 2,
    ylim = c(0,1),
    xlab = expression(alpha),
    ylab = "CC / LC",
    main = main
  )
  
  lines(alpha, LC,
        col = "blue",
        lwd = 2)
  
  legend(
    "topleft",
    legend = c("Concentration Curve",
               "Lorenz Curve"),
    col = c("red","blue"),
    lwd = 2,
    bty = "n"
  )
}


#plot CC and LC curve for GAM1
plot_CC_LC(
  Y = D_hat$claims,
  pred = D_hat$pred_gam1,
  main = "GAM1"
)


#plot CC and LC curve for GAM2
plot_CC_LC(
  Y = D_hat$claims,
  pred = D_hat$pred_gam2,
  main = "GAM2"
)



#plot CC and LC curve for GLM
plot_CC_LC(
  Y = D_hat$claims,
  pred = D_hat$pred_glm1,
  main = "GLM"
)


#plot CC and LC curve for GAM1_noise
plot_CC_LC(
  Y = D_hat$claims,
  pred = D_hat$pred_gam1_noise,
  main = "GAM1_Noise"
)


#plot CC and LC curve for GAM2_noise
plot_CC_LC(
  Y = D_hat$claims,
  pred = D_hat$pred_gam2_noise,
  main = "GAM2_Noise"
)






# Section 5.4: Auto-calibrating GAM1

# Step 1: get GAM1 and GAM1_noise predictions on D2
# D2 is used to implement auto-calibration
D2$pred_gam1 <- predict(gam1, newdata = D2, type = "response")
D2$pred_gam1_noise <- predict(gam1_noise, newdata = D2, type = "response")

# Step 2: build function to choose delta
local_poisson_lcv <- function(Y, expo, pred, delta_grid) {
  
  ord <- order(pred)
  Y <- Y[ord]
  expo <- expo[ord]
  pred <- pred[ord]
  
  n <- length(Y)
  idx <- seq_len(n)
  
  # Add 0 at the beginning so that the sum from left to right can be computed by:
  # cy[right + 1] - cy[left].
  cy <- c(0, cumsum(Y))
  ce <- c(0, cumsum(expo))
  
  out <- data.frame(
    delta = delta_grid,
    deviance = NA_real_
  )
  
  for (j in seq_along(delta_grid)) {
    
    delta <- delta_grid[j]
    
    # Convert delta from a fraction into the number of observations in each neighborhood.
    # ceiling() rounds up because k must be an integer.
    k <- ceiling(delta * n)
    #k <- max(k, 20)
    
    half <- floor((k - 1) / 2)
    
    left <- idx - half
    # The left boundary cannot be smaller than 1.
    # For early observations, idx - half may be negative or zero.
    left <- pmax(left, 1)
    # The left boundary cannot be too large.
    # If the window has length k, the largest possible left boundary is n - k + 1,
    # otherwise right = left + k - 1 would exceed n.
    left <- pmin(left, n - k + 1)
    
    # Right boundary of each local window.
    # Since the window length is k, right = left + k - 1.
    right <- left + k - 1
    
    sum_y <- cy[right + 1] - cy[left]
    sum_e <- ce[right + 1] - ce[left]
    
    # leave-one-out version
    sum_y_loo <- sum_y - Y
    sum_e_loo <- sum_e - expo
    
    local_freq <- sum_y_loo / sum_e_loo
    mu_hat <- expo * local_freq
    mu_hat <- pmax(mu_hat, 1e-12)
    
    #Poisson Deviance
    dev <- 2 * sum(
      ifelse(Y == 0, 0, Y * log(Y / mu_hat)) - (Y - mu_hat)
    )
    
    out$deviance[j] <- dev
    
    cat("delta =", delta, "deviance =", dev, "\n")
    
    #clean the storage
    gc()
  }
  
  out
}

#get delta for both gam1 and gam1_noise model
delta_grid <- seq(0.02, 0.06, by = 0.001)

cv_gam1 <- local_poisson_lcv(
  Y = D2$claims,
  expo = D2$expo,
  pred = D2$pred_gam1,
  delta_grid = delta_grid
)
cv_gam1_noise <- local_poisson_lcv(
  Y = D2$claims,
  expo = D2$expo,
  pred = D2$pred_gam1_noise,
  delta_grid = delta_grid
)


delta_opt_gam1 <- cv_gam1$delta[which.min(cv_gam1$deviance)]
delta_opt_gam1_noise <- cv_gam1_noise$delta[which.min(cv_gam1_noise$deviance)]

delta_opt_gam1
delta_opt_gam1_noise

plot(cv_gam1$delta, cv_gam1$deviance, type = "l",
     xlab = expression(delta),
     ylab = "Leave-one-out Poisson deviance",
     main = "LCV for GAM1")

abline(v = delta_opt_gam1, lty = 2)



plot(cv_gam1_noise$delta, cv_gam1_noise$deviance, type = "l",
     xlab = expression(delta),
     ylab = "Leave-one-out Poisson deviance",
     main = "LCV for GAM1_Noise")

abline(v = delta_opt_gam1_noise, lty = 2)





# Step 3: define local balance correction function
# For a new predicted value m0, find the closest delta fraction of D2 predictions.
# Then set corrected prediction = average observed claims / average exposure
# Finally multiply by the new policy's exposure.

local_balance_predict <- function(m_new, expo_new, m_calib, y_calib, expo_calib, delta = 0.037) {
  
  n_calib <- length(m_calib)
  k <- ceiling(delta * n_calib)
  
  corrected <- numeric(length(m_new))
  
  for (j in seq_along(m_new)) {
    
    # find closest predictions in D2
    dist <- abs(m_calib - m_new[j])
    idx <- order(dist)[1:k]
    
    # local annual frequency
    local_freq <- sum(y_calib[idx]) / sum(expo_calib[idx])
    
    # corrected expected claim count
    corrected[j] <- expo_new[j] * local_freq
  }
  
  corrected
}





# Step 4: apply correction to validation set D_hat
D_hat$pred_gam1_auto <- local_balance_predict(
  m_new = D_hat$pred_gam1,
  expo_new = D_hat$expo,
  m_calib = D2$pred_gam1,
  y_calib = D2$claims,
  expo_calib = D2$expo,
  delta = delta_opt_gam1
)


# Step 4: apply correction to validation set D_hat
D_hat$pred_gam1_noise_auto <- local_balance_predict(
  m_new = D_hat$pred_gam1_noise,
  expo_new = D_hat$expo,
  m_calib = D2$pred_gam1_noise,
  y_calib = D2$claims,
  expo_calib = D2$expo,
  delta = delta_opt_gam1_noise
)



# Step 5: draw CC/LC curves for corrected GAM1
plot_CC_LC(
  Y = D_hat$claims,
  pred = D_hat$pred_gam1_auto,
  main = "GAM1 after Auto-calibration"
)

# Step 5: draw CC/LC curves for corrected GAM1 Noise
plot_CC_LC(
  Y = D_hat$claims,
  pred = D_hat$pred_gam1_noise_auto,
  main = "GAM1 Noise after Auto-calibration"
)


# Step 6: compute p-values again for GAM1
p_equal_gam1_auto <- p_equal_test(
  D_hat$claims,
  D_hat$pred_gam1_auto,
  B = 500
)

p_auto_gam1_auto <- p_auto_test(
  D_hat$claims,
  D_hat$pred_gam1_auto,
  B = 500
)

p_equal_gam1_auto
p_auto_gam1_auto

# Step 6: compute p-values again for GAM1 Noise
p_equal_gam1_noise_auto <- p_equal_test(
  D_hat$claims,
  D_hat$pred_gam1_noise_auto,
  B = 500
)

p_auto_gam1_noise_auto <- p_auto_test(
  D_hat$claims,
  D_hat$pred_gam1_noise_auto,
  B = 500
)

p_equal_gam1_noise_auto
p_auto_gam1_noise_auto















