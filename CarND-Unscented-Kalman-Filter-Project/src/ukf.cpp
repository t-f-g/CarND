#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, lidar measurements will be ignored (except during init)
  use_lidar_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  //set state dimension
  n_x_ = 5;
  //set augmented dimension
  n_aug_ = n_x_ + 2;
  //define spreading parameter
  lambda_ = 3 - n_aug_;
  n_sigma_ = (2 * n_aug_) + 1; //2n+1 sigma points
  n_z_lidar_ = 2;
  n_z_radar_ = 3;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  //std_a_ = 30;
  std_a_ = 0.65;

  // Process noise standard deviation yaw acceleration in rad/s^2
  //std_yawdd_ = 30;
  std_yawdd_ = 1.;

  // Laser measurement noise standard deviation position1 in m
  std_lidar_px_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_lidar_py_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radar_r_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radar_phi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radar_rd_ = 0.3;

  is_initialized_ = false;

  // initial state vector
  x_ = VectorXd::Zero(n_x_ );
  P_ = MatrixXd::Identity(n_x_, n_x_);
  x_aug_ = VectorXd::Zero(n_aug_);
  P_aug_ = MatrixXd::Zero(n_aug_, n_aug_);
  Q_ = MatrixXd(n_aug_ - n_x_, n_aug_ - n_x_);
  sigma_ = MatrixXd(n_x_, n_sigma_);
  sigma_aug_ = MatrixXd(n_aug_, n_sigma_);

  weights_ = VectorXd(n_sigma_);
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }

  Q_ << pow(std_a_, 2),                  0,
                     0, pow(std_yawdd_, 2);

  R_lidar_ = MatrixXd::Zero(n_z_lidar_, n_z_lidar_);
  R_lidar_ << pow(std_lidar_px_, 2),                     0,
                                  0, pow(std_lidar_py_, 2);

  R_radar_ = MatrixXd::Zero(n_z_radar_, n_z_radar_);
  R_radar_ << pow(std_radar_r_, 2),                      0,                     0,
                                 0, pow(std_radar_phi_, 2),                     0,
                                 0,                      0, pow(std_radar_rd_, 2);
  NIS_radar_ = VectorXd::Zero(n_x_ );
  NIS_lidar_ = VectorXd::Zero(n_x_ );
}

UKF::~UKF() {}

float UKF::AngleNormalization(float angle) {
  //while (angle > M_PI) angle -= 2. * M_PI;
  //while (angle < -M_PI) angle += 2. * M_PI;
  // or
  angle = std::fmod(angle, 2. * M_PI);
  return angle;
}

void UKF::Init(MeasurementPackage meas_package) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
      previous_timestamp_ = meas_package.timestamp_;
      is_initialized_ = true;
      if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
          float rho = meas_package.raw_measurements_[0];
          float phi = meas_package.raw_measurements_[1];
          float rho_dot = meas_package.raw_measurements_[2];
          x_(0) = rho * cos(phi);
          x_(1) = rho * sin(phi);
      } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
          x_(0) = meas_package.raw_measurements_[0];
          x_(1) = meas_package.raw_measurements_[1];
      }
  }
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/

   if (!is_initialized_) {
     Init(meas_package);
     cout << "Initialized." << endl;
     return;
   }

   /*****************************************************************************
    *  Calculate elapsed time and update saved timestamp for next calculation
    ****************************************************************************/

   float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
   previous_timestamp_ = meas_package.timestamp_;

  /*****************************************************************************
   *  Prediction (Common)
   ****************************************************************************/

  Prediction(dt);

  /*****************************************************************************
   *  Predict and Update (Sensor specific)
   ****************************************************************************/

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Radar updates
      PredictAndUpdateRadar(meas_package);
  } else {
      // Lidar updates
      PredictAndUpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double dt) {
  /*****************************************************************************
   *  Augment Sigma points
   * from Lesson "Augmentation Assignment 2"
   ****************************************************************************/

  // Create augmented mean state
  x_aug_ << x_, 0, 0;

  P_aug_.fill(0.0); //clear augmented sigma points
  P_aug_.topLeftCorner(n_x_, n_x_) = P_;
  P_aug_.bottomRightCorner(n_aug_ - n_x_, n_aug_ - n_x_) = Q_;

  // Create square root matrix
  MatrixXd L = P_aug_.llt().matrixL();

  // Create augmented sigma points
  sigma_aug_.col(0) = x_aug_;

  for (int i = 0; i < n_aug_; i++)
  {
    sigma_aug_.col(i + 1)          = x_aug_ + sqrt(lambda_ + n_aug_) * L.col(i);
    sigma_aug_.col(i + 1 + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  /*****************************************************************************
   *  Predict Sigma points
   *  from lesson "Sigma Point Prediction Assignment 2"
   ****************************************************************************/
   for (int i = 0; i < n_sigma_; i++)
   {
     // Extract values for better readibility
     double p_x = sigma_aug_(0, i);
     double p_y = sigma_aug_(1, i);
     double v = sigma_aug_(2, i);
     double yaw = sigma_aug_(3, i);
     double yawd = sigma_aug_(4, i);
     double nu_a = sigma_aug_(5, i);
     double nu_yawdd = sigma_aug_(6, i);

     // Predicted state values
     double px_p, py_p;

     // Avoid division by zero
     if (fabs(yawd) > 0.001) {
       px_p = p_x + (v / yawd) * (sin(yaw + (yawd * dt)) - sin(yaw));
       py_p = p_y + (v / yawd) * (cos(yaw) - cos(yaw + (yawd * dt)));
     }
     else {
       px_p = p_x + (v * dt * cos(yaw));
       py_p = p_y + (v * dt * sin(yaw));
     }

     double v_p = v;
     double yaw_p = yaw + (yawd * dt);
     double yawd_p = yawd;

     // Add noise
     px_p = px_p + (0.5 * nu_a * pow(dt, 2) * cos(yaw));
     py_p = py_p + (0.5 * nu_a * pow(dt, 2) * sin(yaw));
     v_p = v_p + (nu_a * dt);

     yaw_p = yaw_p + (0.5 * nu_yawdd * pow(dt, 2));
     yawd_p = yawd_p + (nu_yawdd * dt);

     // Write predicted sigma point into right column
     sigma_(0, i) = px_p;
     sigma_(1, i) = py_p;
     sigma_(2, i) = v_p;
     sigma_(3, i) = yaw_p;
     sigma_(4, i) = yawd_p;
   }

  /*****************************************************************************
   *  Predict Mean and covariance
   * from Lesson "Predicted Mean and Covariance Assignment 2"
   ****************************************************************************/

   x_.fill(0.0);
   for (int i = 0; i < n_sigma_; i++) {  //iterate over sigma points
     x_ = x_ + (weights_(i) * sigma_.col(i));
   }

   P_.fill(0.0);
   for (int i = 0; i < n_sigma_; i++) {  //iterate over sigma points
     VectorXd x_diff = sigma_.col(i) - x_;
     x_diff(3) = AngleNormalization(x_diff(3));
     P_ = P_ + (weights_(i) * x_diff * x_diff.transpose());
   }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::PredictAndUpdateLidar(MeasurementPackage meas_package) {
  Z_sigma_ = MatrixXd(n_z_lidar_, n_sigma_);
  VectorXd z = meas_package.raw_measurements_;

  // Copy to measurement space
  for (int i = 0; i < n_sigma_; i++) {  //2n+1 simga points
    Z_sigma_(0, i) = sigma_(0, i);
    Z_sigma_(1, i) = sigma_(1, i);
  }

  // Predict mean
  z_pred_ = VectorXd(n_z_lidar_);
  z_pred_.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {
    z_pred_ = z_pred_ + (weights_(i) * Z_sigma_.col(i));
  }

  // Calculate covariance matrix S
  S_ = MatrixXd(n_z_lidar_, n_z_lidar_);
  S_.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {  //2n+1 simga points
    VectorXd z_diff = Z_sigma_.col(i) - z_pred_;
    z_diff(1) = AngleNormalization(z_diff(1));
    S_ = S_ + (weights_(i) * z_diff * z_diff.transpose());
  }

  // Add noise
  S_ = S_ + R_lidar_;

  /**
  from Lesson "UKF Update Assignment 2"
  */

  // Calculate cross correlation matrix Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_lidar_);
  Tc.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {  //2n+1 simga points
    VectorXd z_diff = Z_sigma_.col(i) - z_pred_;
    VectorXd x_diff = sigma_.col(i) - x_;
    Tc = Tc + (weights_(i) * x_diff * z_diff.transpose());
  }

  // Calculate Kalman Gain;
  MatrixXd K = Tc * S_.inverse();

  VectorXd z_diff = z - z_pred_;
  z_diff(1) = AngleNormalization(z_diff(1));

  // Update covariance and state matrix
  x_ = x_ + (K * z_diff);
  P_ = P_ - (K * S_ * K.transpose());
  NIS_lidar_ = S_.inverse() * z_diff * z_diff.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::PredictAndUpdateRadar(MeasurementPackage meas_package) {
  Z_sigma_ = MatrixXd(n_z_radar_, n_sigma_);
  VectorXd z = meas_package.raw_measurements_;

  // Transform sigma points to measurement space
  for (int i = 0; i < n_sigma_; i++) {  //2n+1 simga points
    // Extract values for better readibility
    double p_x = sigma_(0, i);
    double p_y = sigma_(1, i);
    double v  = sigma_(2, i);
    double yaw = sigma_(3, i);
    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;
    double r = sqrt((p_x * p_x) + (p_y * p_y));
    // Avoid division by zero
    if (fabs(r) < 0.00001) r = 0.00001;
    Z_sigma_(0, i) = r;
    Z_sigma_(1, i) = atan2(p_y, p_x);
    Z_sigma_(2, i) = ((p_x * v1) + (p_y * v2)) / r;
  }

  // Predict mean
  z_pred_ = VectorXd(n_z_radar_);
  z_pred_.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {
    z_pred_ = z_pred_ + (weights_(i) * Z_sigma_.col(i));
  }

  // Calculate covariance matrix S
  S_ = MatrixXd(n_z_radar_, n_z_radar_);
  S_.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {  //2n+1 simga points
    VectorXd z_diff = Z_sigma_.col(i) - z_pred_;
    z_diff(1) = AngleNormalization(z_diff(1));
    S_ = S_ + (weights_(i) * z_diff * z_diff.transpose());
  }

  // Add noise
  S_ = S_ + R_radar_;

  /**
  from Lesson "UKF Update Assignment 2"
  */

  // Calculate cross correlation matrix Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);
  Tc.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {  //2n+1 simga points
    VectorXd z_diff = Z_sigma_.col(i) - z_pred_;
    z_diff(1) = AngleNormalization(z_diff(1));
    VectorXd x_diff = sigma_.col(i) - x_;
    x_diff(3) = AngleNormalization(x_diff(3));
    Tc = Tc + (weights_(i) * x_diff * z_diff.transpose());
  }

  // Calculate Kalman Gain
  MatrixXd K = Tc * S_.inverse();

  VectorXd z_diff = z - z_pred_;
  z_diff(1) = AngleNormalization(z_diff(1));

  // Update covariance and state matrix
  x_ = x_ + (K * z_diff);
  P_ = P_ - (K * S_ * K.transpose());
  NIS_radar_ =  S_.inverse() * z_diff * z_diff.transpose();
}
