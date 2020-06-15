// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Include file for Occnet Class.
// // This is a stub for now.
// Class definition
#ifndef OCCNET_H
#define OCCNET_H

class OccNet {
 public:
  OccNet(void);

  void SetShapeEmbedding(const RNScalar* embedding);
  RNScalar Evaluate(const R3Point& position);

  virtual int ReadFile(const char* filename);

 private:
  std::vector<float> sample_resize_fc_weights;
  std::vector<float> sample_resize_fc_biases;
  std::vector<float> fc1_weights;
  std::vector<float> fc1_biases;
  std::vector<float> fc2_weights;
  std::vector<float> fc2_biases;
  float cbn1_running_mean;
  float cbn1_running_variance;
  float cbn2_running_mean;
  float cbn2_running_variance;
  float cbnf_running_mean;
  float cbnf_running_variance;
  std::vector<float> final_activation_weights;
  std::vector<float> final_activation_bias;
};

inline void SetShapeEmbedding(const RNScalar* embedding) { return; }

inline RNScalar OccNet::Evaluate(const R3Point& position) { return 0.0; }

#endif  // OCCNET_H
