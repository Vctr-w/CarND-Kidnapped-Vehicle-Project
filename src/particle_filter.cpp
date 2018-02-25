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
	num_particles = 50;

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	normal_distribution<double> dist_x(x, std_x);	
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;

		particles.push_back(p);
		weights.push_back(1);
	}

	is_initialized = true;
	return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	normal_distribution<double> dist_x(0, std_x);	
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	for (int i = 0; i < num_particles; i++) {
		double theta = particles[i].theta;

		if (yaw_rate != 0) {
			particles[i].x += (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
			particles[i].y += (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
		}
		else {
			particles[i].x += velocity * cos(theta) * delta_t;
			particles[i].y += velocity * sin(theta) * delta_t;
		}

		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += yaw_rate * delta_t + dist_theta(gen);
	}
}

std::vector<LandmarkObs> ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
//void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	vector<LandmarkObs> assoc;

	for (int i = 0; i < observations.size(); i++) {
		double min_error = numeric_limits <double>::max();

		LandmarkObs l;

		for (int j = 0; j < predicted.size(); j++) {
			double error = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if (error < min_error) {
				min_error = error;
				l.x = predicted[j].x;
				l.y = predicted[j].y;
			}
		}

		assoc.push_back(l);
	}

	return assoc;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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

	double std_x = std_landmark[0];
	double std_y = std_landmark[1];

	double weights_sum = 0;

	for (int i = 0; i < num_particles; i++) {
		// Get list of in range landmarks

		vector <LandmarkObs> in_range_landmarks;

		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			double distance = dist(particles[i].x, particles[i].y, 
				map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);

			if (distance < sensor_range) {
				LandmarkObs l;

				l.x = map_landmarks.landmark_list[j].x_f;
				l.y = map_landmarks.landmark_list[j].y_f;

				in_range_landmarks.push_back(l);
			}
		}

		// Transform observations to map

		vector <LandmarkObs> transf_observation;

		for (int j = 0; j < observations.size(); j++) {
			LandmarkObs l;

			double x_p = particles[i].x;
			double y_p = particles[i].y;
			double theta_p = particles[i].theta;
			double x_c = observations[j].x;
			double y_c = observations[j].y;

			l.x = x_p + (cos(theta_p) * x_c) - (sin(theta_p) * y_c);
			l.y = y_p + (sin(theta_p) * x_c) + (cos(theta_p) * y_c);

			transf_observation.push_back(l);
		}

		vector <LandmarkObs> associated_obs = dataAssociation(in_range_landmarks, transf_observation);

		// Update weights

		particles[i].weight = 1;
		weights[i] = 1;

		for (int j = 0; j < associated_obs.size(); j++) {
			double dx = associated_obs[j].x - transf_observation[j].x;
			double dy = associated_obs[j].y - transf_observation[j].y;
			
			double gauss_norm = (1 / (2 * M_PI * std_x * std_y));
			double exponent = ((dx * dx) / (2 * std_x * std_x)) + 
				((dy * dy) / (2 * std_y * std_y));

			particles[i].weight *= gauss_norm * exp(-exponent);
			weights[i] *= gauss_norm * exp(-exponent);
		}

		weights_sum += particles[i].weight;
	}

	for (int i = 0; i < num_particles; i++) {
		particles[i].weight /= weights_sum;
		weights[i] /= weights_sum;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution <int> distribution(weights.begin(), weights.end());

	vector <Particle> resampled_particles;

	for (int i = 0; i < num_particles; i++) {
		resampled_particles.push_back(particles[distribution(gen)]);
	}

	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
