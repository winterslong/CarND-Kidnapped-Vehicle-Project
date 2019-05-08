/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

static std::default_random_engine  gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles

  double std_x = std[0]; 
  double std_y = std[1]; 
  double std_theta = std[2];
  
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);
  
  for (int i = 0; i < num_particles; ++i)
  {
    Particle  new_item;
    new_item.id = i;
    new_item.weight = 1.0f;
    new_item.x = dist_x(gen);
    new_item.y = dist_y(gen);
    new_item.theta = dist_theta(gen);

    particles.push_back(new_item);
  }

  // Initialize the weights
  weights.assign(num_particles, 1.0f);

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  double std_x = std_pos[0]; 
  double std_y = std_pos[1]; 
  double std_theta = std_pos[2];  
  normal_distribution<double> dist_x(0, std_x);
  normal_distribution<double> dist_y(0, std_y);
  normal_distribution<double> dist_theta(0, std_theta);
  
  for (int i = 0; i < num_particles; ++i) {
    if (fabs(yaw_rate) > 0.0001f){
      // Updating x, y and the yaw angle when the yaw rate is not equal to zero:
      double theta0 = particles[i].theta;
      particles[i].x += (velocity/yaw_rate) * (sin(theta0 + yaw_rate*delta_t) - sin(theta0));
      particles[i].y += (velocity/yaw_rate) * (-cos(theta0 + yaw_rate*delta_t) + cos(theta0));
      particles[i].theta += yaw_rate * delta_t;
    } else {
      // When yaw_rate is equal to zero
      particles[i].x += (velocity * delta_t) * cos(particles[i].theta);
      particles[i].y += (velocity * delta_t) * sin(particles[i].theta);
    }

    // Add random noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  for (int i = 0; i < observations.size(); i++) {
    LandmarkObs obs = observations[i];
  
    // Init minimum distance to maximum
    double min_dist = std::numeric_limits<double>::max();
    
    int map_id = -1;    
    for (unsigned int j = 0; j < predicted.size(); j++) {
      LandmarkObs landmark = predicted[j];
      
      double cur_dist = dist(obs.x, obs.y, landmark.x, landmark.y);
  
      // Find the predicted landmark nearest the current observed landmark
      if (cur_dist < min_dist) {
        min_dist = cur_dist;
        map_id = landmark.id;
      }
    }
  
    observations[i].id = map_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
    
  for (int i = 0; i < num_particles; i++) {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    /* The landmarks within the sensor range of the particle.      
     */
    vector<LandmarkObs> effected_landmarks;

    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;

      if ((fabs(lm_x - p_x) <= sensor_range) 
            && (fabs(lm_y - p_y) <= sensor_range)) {
        effected_landmarks.push_back(LandmarkObs{lm_id, lm_x, lm_y });
      }
    }

    /* Transform observations from the car coordinate system into map coordinates
      */
    vector<LandmarkObs> map_observations;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double m_x = p_x + cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y;
      double m_y = p_y + sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y;
      map_observations.push_back(LandmarkObs{ observations[j].id, m_x, m_y });
    }

    /* Data association for the effected_landmarks and map observations
      */
    dataAssociation(effected_landmarks, map_observations);

    // Re-init weight
    particles[i].weight = 1.0f;

    for (unsigned int j = 0; j < map_observations.size(); j++) {
      double obs_x, obs_y, mu_lm_x, mu_lm_y;
      
      obs_x = map_observations[j].x;
      obs_y = map_observations[j].y;

      int associated_prediction = map_observations[j].id;

      // Get the x,y coordinates of the prediction associated with the current observation
      for (unsigned int k = 0; k < effected_landmarks.size(); k++) {
        if (effected_landmarks[k].id == associated_prediction) {
          mu_lm_x = effected_landmarks[k].x;
          mu_lm_y = effected_landmarks[k].y;
          break;
        }
      }

      // Calculate weight for this observation with multivariate Gaussian
      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];
      double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);        
      double exponent = (pow(obs_x - mu_lm_x, 2) / (2*pow(sig_x, 2))) 
                       + (pow(obs_y - mu_lm_y, 2) / (2*pow(sig_y, 2)));
    
      double obs_weight = gauss_norm * exp(-exponent) ;

      // Multiply all the calculated measurement probabilities to get the final weight
      particles[i].weight *= obs_weight;
    }
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */    
  vector<Particle> new_particles;

  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // Generate random starting index for resampling wheel
  std::uniform_int_distribution<int> uni_int_dist(0, num_particles-1);
  auto index = uni_int_dist(gen);

  double max_weight = *max_element(weights.begin(), weights.end());

  // Uniform random distribution [0.0, max_weight)
  std::uniform_real_distribution<double> uni_real_dist(0.0, max_weight);

  double beta = 0.0;

  // Resampling Wheel  
  for (int i = 0; i < num_particles; i++) {
    beta += uni_real_dist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
