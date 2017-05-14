#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, lidar measurements will be ignored (except for init)
  bool use_lidar_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;
  MatrixXd Q_;
  MatrixXd S_;

  ///* predicted sigma points matrix
  MatrixXd sigma_;
  MatrixXd sigma_aug_;
  MatrixXd z_pred_;
  MatrixXd Z_sigma_;

  MatrixXd R_lidar_;
  MatrixXd R_radar_;

  MatrixXd x_aug_;
  MatrixXd P_aug_;
  ///* time when the state is true, in us
  long long previous_timestamp_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Lidar measurement noise standard deviation position1 in m
  double std_lidar_px_;

  ///* Lidar measurement noise standard deviation position2 in m
  double std_lidar_py_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radar_r_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radar_phi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radar_rd_ ;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  int n_x_;
  int n_z_lidar_;
  int n_z_radar_;

  ///* Augmented state dimension
  int n_aug_;

  ///* number of sigma points
  int n_sigma_;

  ///* Sigma point spreading parameter
  double lambda_;

  ///* the current NIS for radar
  MatrixXd NIS_radar_;

  ///* the current NIS for lidar
  MatrixXd NIS_lidar_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * Initialization
   * @param meas_package The latest measurement data of either radar or lidar
   */
  void Init(MeasurementPackage meas_package);

  /**
   * AngleNormalization
   * @param angle angle to be normalized
   * returns normalized angle
   */
  float AngleNormalization(float angle);

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or lidar
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param dt Time between k and k+1 in s
   */
  void Prediction(double dt);

  /**
   * Updates the state and the state covariance matrix using a lidar measurement
   * @param meas_package The measurement at k+1
   */
  void PredictAndUpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void PredictAndUpdateRadar(MeasurementPackage meas_package);
};

#endif /* UKF_H */
