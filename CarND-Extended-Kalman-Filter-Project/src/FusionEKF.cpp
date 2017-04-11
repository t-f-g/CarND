#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // Initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd::Zero(3, 4);

    // Measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
                0, 0.0225;

    // Measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
                0, 0.0009, 0,
                0, 0, 0.09;

    // From "sensor-fusion-ekf-reference.pdf" formula 42:
    H_laser_ << 1, 0, 0, 0,
                0, 1, 0, 0;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        /**
          * Initialize the state ekf_.x_ with the first measurement.
          * Create the covariance matrix.
        */
        // first measurement
        cout << "EKF: " << endl;

    	  // Initialize Vectors and Matrices
        ekf_.x_ = VectorXd(4);
        ekf_.P_ = MatrixXd(4, 4);
        ekf_.F_ = MatrixXd::Zero(4, 4);
        ekf_.Q_ = MatrixXd::Zero(4, 4);

    	  // Initialize state vector
        ekf_.x_ << 1, 1, 1, 1;

        // Initialize state covariance matrix
        ekf_.P_ << 1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1;

    	  // Initialize prior timestamp
        previous_timestamp_ = measurement_pack.timestamp_;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
              /**
              Convert radar from polar to cartesian coordinates and initialize state.
              */
              float rho = measurement_pack.raw_measurements_(0);
              float phi = measurement_pack.raw_measurements_(1);
              float rho_dot = measurement_pack.raw_measurements_(2);

              // Math is Fun: https://www.mathsisfun.com/polar-cartesian-coordinates.html
              float px = rho * cos(phi);
              float py = rho * sin(phi);
              float vx = rho_dot * cos(phi);
              float vy = rho_dot * sin(phi);

              ekf_.x_ << px,
        	               py,
        				         vx,
        				         vy;
          }
          else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
              /**
              Initialize state.
              */
              float px = measurement_pack.raw_measurements_(0);
              float py = measurement_pack.raw_measurements_(1);
              // velocity unknown

              ekf_.x_ << px,
        	               py,
        				         0.0,
        				         0.0;
        }

        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    /**
       * Update the state transition matrix F according to the new elapsed time.
        - Time is measured in seconds.
       * Update the process noise covariance matrix.
       * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
     */

    float noise_ax = 9;
    float noise_ay = 9;
    // convert from microseconds to seconds
    float delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    float delta_t_2 = pow(delta_t, 2);
    float delta_t_3 = pow(delta_t, 3);
    float delta_t_4 = pow(delta_t, 4);

    // From "sensor-fusion-ekf-reference.pdf" formula 40:
    ekf_.Q_ <<  delta_t_4/4*noise_ax, 0, delta_t_3/2*noise_ax, 0,
                0, delta_t_4/4*noise_ay, 0, delta_t_3/2*noise_ay,
                delta_t_3/2*noise_ax, 0, delta_t_2*noise_ax, 0,
                0, delta_t_3/2*noise_ay, 0, delta_t_2*noise_ay;

    // From "sensor-fusion-ekf-reference.pdf" formula 21 or 99:
    ekf_.F_ <<  1, 0, delta_t, 0,
                0, 1, 0, delta_t,
                0, 0, 1, 0,
                0, 0, 0, 1;

    // Save timestamp for next call
    previous_timestamp_ = measurement_pack.timestamp_;

    ekf_.Predict();

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    /**
       * Use the sensor type to perform the update step.
       * Update the state and covariance matrices.
     */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        VectorXd radar(3);

        float rho = measurement_pack.raw_measurements_(0);
        float phi = measurement_pack.raw_measurements_(1);
        float rho_dot = measurement_pack.raw_measurements_(2);

        radar << rho, phi, rho_dot;

        Tools tools;
        Hj_ = tools.CalculateJacobian(ekf_.x_);

        ekf_.H_ = Hj_;
        ekf_.R_ = R_radar_;

        ekf_.UpdateEKF(radar);
    } else {
        VectorXd laser(2);

        laser << measurement_pack.raw_measurements_(0), measurement_pack.raw_measurements_(1);

        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;

        ekf_.Update(laser);
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
