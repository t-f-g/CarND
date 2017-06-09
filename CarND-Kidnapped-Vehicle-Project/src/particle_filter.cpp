/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <float.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
  	// Add random Gaussian noise to each particle.
  	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    default_random_engine gen;

    num_particles = 100;
    weights.resize(num_particles);
    particles.resize(num_particles);

    normal_distribution<double> N_x(x, std[0]);
  	normal_distribution<double> N_y(y, std[1]);
  	normal_distribution<double> N_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {
        particles[i].id = i;
        particles[i].x = N_x(gen);
        particles[i].y = N_y(gen);
        particles[i].theta = N_theta(gen);
        particles[i].weight = 1;
        weights[i] = 1;
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;

    for (int i = 0; i < num_particles; i++) {
        if (fabs(yaw_rate) < 1e-9) {
            particles[i].x = particles[i].x + (velocity * delta_t * cos(particles[i].theta));
            particles[i].y = particles[i].y + (velocity * delta_t * sin(particles[i].theta));
        } else {
            particles[i].x = particles[i].x + (velocity / yaw_rate * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta)));
            particles[i].y = particles[i].y + (velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t))));
            particles[i].theta = particles[i].theta + (yaw_rate * delta_t);
        }
        normal_distribution<double> N_x(0, std_pos[0]);
        normal_distribution<double> N_y(0, std_pos[1]);
        normal_distribution<double> N_theta(0, std_pos[2]);
        // Add noise
        particles[i].x = particles[i].x + N_x(gen);
        particles[i].y = particles[i].y + N_y(gen);
        particles[i].theta = particles[i].theta + N_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  	//   observed measurement to this particular landmark.
  	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  	//   implement this method and use it as a helper during the updateWeights phase.

    for (int o = 0; o < observations.size(); o++) {
        double dist_min = DBL_MAX;

        for (int i = 0; i < predicted.size(); i++) {
            double dist_to_o = dist(observations[o].x, observations[o].y, predicted[i].x, predicted[i].y);
            if (dist_to_o < dist_min){
                dist_min = dist_to_o;
                observations[o].id = i;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    // Converted observation
    double x_map = 0.0;
    double y_map = 0.0;
    double w_sum = 0.0;

    for (int i = 0; i < num_particles; i++) {
        // Look at all observations and calculate weights
        for(int o = 0; o < observations.size(); o++) {
            // Convert to the map coordinates
            x_map = observations[o].x * cos(particles[i].theta) - observations[o].y * sin(particles[i].theta) + particles[i].x;
            y_map = observations[o].x * sin(particles[i].theta) + observations[o].y * cos(particles[i].theta) + particles[i].y;

            double dist_to_o = dist(x_map, y_map, particles[i].x, particles[i].y);
            if (dist_to_o < sensor_range) {
                double dist_min = DBL_MAX;
                int dist_min_idx = 0;

                for(int l = 0; l < map_landmarks.landmark_list.size(); l++){
                    double dist_to_l = dist(x_map, y_map, map_landmarks.landmark_list[l].x_f, map_landmarks.landmark_list[l].y_f);
                    if (dist_to_l < dist_min and dist_to_l < sensor_range) {
                        // The closest landmark distance gets set to dist_min_idx when the loop completes
                        dist_min = dist_to_l;
                        dist_min_idx = l;
                    }
                }
                particles[i].weight = particles[i].weight * (1 / (2 * M_PI * std_landmark[0] * std_landmark[1])) * exp(-((pow(x_map - map_landmarks.landmark_list[dist_min_idx].x_f, 2) / pow(std_landmark[0], 2)) + (pow(y_map - map_landmarks.landmark_list[dist_min_idx].y_f, 2) / pow(std_landmark[1], 2))) / 2);
            }
        }
        // Accumulate weights for use in Normalization below
        w_sum = w_sum + particles[i].weight;
    }

    // Normalize
    for (int i = 0; i < num_particles; i++){
        particles[i].weight = particles[i].weight / (w_sum * (2 * M_PI * std_landmark[0] * std_landmark[1]));
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  default_random_engine gen;
  discrete_distribution<int> distribution(weights.begin(), weights.end());

  vector<Particle> resample_particles;

  for (int i = 0; i < num_particles; i++) {
      resample_particles.push_back(particles[distribution(gen)]);
  }

  particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  	vector<int> v = best.associations;
  	stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
  	vector<double> v = best.sense_x;
  	stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{
  	vector<double> v = best.sense_y;
  	stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
