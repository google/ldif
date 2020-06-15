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

// Source file for 3D quadric class

// Include files

namespace gaps {};
using namespace gaps;  // NOLINT(build/namespaces)

#include "R3Shapes/R3Shapes.h"

#include "R3Quadric.h"

// Member functions

R3Quadric::R3Quadric(void)
    : matrix(R4null_matrix),
      support(R3xyz_coordinate_system, R3zero_vector),
      features(NULL),
      nfeatures(0),
      symmetry(0),
      flags(R3_QUADRIC_MATRIX_IS_CONSTANT) {}

R3Quadric::R3Quadric(const R3Quadric& quadric)
    : matrix(quadric.matrix),
      support(quadric.support),
      features(NULL),
      nfeatures(quadric.nfeatures),
      symmetry(quadric.symmetry),
      flags(quadric.flags) {
  // Copy features
  if (quadric.features && (quadric.nfeatures > 0)) {
    this->features = new RNScalar[quadric.nfeatures];
    for (int i = 0; i < quadric.nfeatures; i++) {
      this->features[i] = quadric.features[i];
    }
  }
}

R3Quadric::R3Quadric(const R4Matrix& matrix, const R3Ellipsoid& support,
                     int symmetry, RNScalar* features, int nfeatures)
    : matrix(R4null_matrix),
      support(R3xyz_coordinate_system, R3zero_vector),
      features(NULL),
      nfeatures(0),
      symmetry(symmetry),
      flags(R3_QUADRIC_MATRIX_IS_CONSTANT) {
  // Set stuff
  SetMatrix(matrix);
  SetSupport(support);

  // Copy features
  if (features && (nfeatures > 0)) {
    this->nfeatures = nfeatures;
    this->features = new RNScalar[nfeatures];
    for (int i = 0; i < nfeatures; i++) {
      this->features[i] = features[i];
    }
  }
}

R3Quadric::~R3Quadric(void) {
  // Delete features
  if (features) delete[] features;
}

void R3Quadric::Empty(void) {
  // Reset everything
  SetMatrix(R4null_matrix);
  support.Empty();
  symmetry = 0;
  if (features) delete[] features;
  features = NULL;
  nfeatures = 0;
}

void R3Quadric::SetMatrix(const R4Matrix& matrix) {
  // Set matrix
  this->matrix = matrix;

  // Check if matrix has non-constant values
  flags.Add(R3_QUADRIC_MATRIX_IS_CONSTANT);
  for (int i0 = 0; i0 < 4; i0++) {
    for (int i1 = 0; i1 < 4; i1++) {
      if (matrix[i0][i1] != 0) {
        if ((i0 != 3) && (i1 != 3)) {
          flags.Remove(R3_QUADRIC_MATRIX_IS_CONSTANT);
          break;
        }
      }
    }
  }
}

void R3Quadric::SetSupport(const R3Ellipsoid& support) {
  // Set support region
  this->support = support;
}

void R3Quadric::SetSymmetry(int symmetry) {
  // Set symmetry type
  this->symmetry = symmetry;
}

void R3Quadric::SetFeatures(const RNScalar* features, int nfeatures) {
  // Delete previous features
  if (this->features) {
    delete[] this->features;
    this->features = NULL;
    this->nfeatures = 0;
  }

  // Assign new features
  if (features && (nfeatures > 0)) {
    this->nfeatures = nfeatures;
    this->features = new RNScalar[nfeatures];
    for (int i = 0; i < nfeatures; i++) {
      this->features[i] = features[i];
    }
  }
}

void R3Quadric::Translate(const R3Vector& vector) {
  // Translate support
  support.Translate(vector);
}

void R3Quadric::Transform(const R3Transformation& transformation) {
  // Transform support
  support.Transform(transformation);
}

RNScalar R3Quadric::WeightedValue(const R3Point& position) const {
  // Return weighted value
  return UnweightedValue(position) * Weight(position);
}

RNScalar R3Quadric::UnweightedValue(const R3Point& position) const {
  // Check matrix
  if (flags[R3_QUADRIC_MATRIX_IS_CONSTANT]) {
    // Return constant term of matrix
    return ConstantTerm();
  } else {
    // Get (maybe transformed) input position
    R3Point input = position;
    if (symmetry == 1) input[2] = -input[2];

    // Compute the position in the coordinate frame of the support
    const R3Vector& v = input - Center();
    const R3Triad& axes = support.CoordSystem().Axes();
    R3Point p(v.Dot(axes[0]), v.Dot(axes[1]), v.Dot(axes[2]));

    // Compute value of quadric implicit
    const R4Matrix& m = matrix;
    RNScalar a = p[0] * m[0][0] + p[1] * m[1][0] + p[2] * m[2][0] + m[3][0];
    RNScalar b = p[0] * m[0][1] + p[1] * m[1][1] + p[2] * m[2][1] + m[3][1];
    RNScalar c = p[0] * m[0][2] + p[1] * m[1][2] + p[2] * m[2][2] + m[3][2];
    RNScalar d = p[0] * m[0][3] + p[1] * m[1][3] + p[2] * m[2][3] + m[3][3];
    return a * p[0] + b * p[1] + c * p[2] + d;
  }
}

RNScalar R3Quadric::Weight(const R3Point& position) const {
  // Get (maybe transformed) input position
  R3Point input = position;
  if (symmetry == 1) input[2] = -input[2];

  // Compute the position in the coordinate frame of the support
  const R3Vector& v = input - Center();
  const R3Triad& axes = support.CoordSystem().Axes();
  R3Point p(v.Dot(axes[0]), v.Dot(axes[1]), v.Dot(axes[2]));

  // Compute weight
  RNScalar weight = 0;
  R3Vector r = support.Radii();
  weight += p[0] * p[0] / (-2.0 * r[0] * r[0]);
  weight += p[1] * p[1] / (-2.0 * r[1] * r[1]);
  weight += p[2] * p[2] / (-2.0 * r[2] * r[2]);
  weight = exp(weight);

  // Return weight
  return weight;
}

const RNLength R3Quadric::Radius(int dim,
                                 RNScalar max_weighted_magnitude) const {
  // Return radius(dim) outside which abs(weighted value) is less than
  // max_weighted_magnitude
  if (!IsMatrixConstant()) return RN_INFINITY;
  RNScalar constant_value = UnweightedValue(R3zero_point);
  RNScalar inflation =
      sqrt(-2.0 * log(max_weighted_magnitude / fabs(constant_value)));
  RNLength radius = inflation * Radius(dim);
  // R3Point query_position = Center() + radius * Axis(dim);
  // RNScalar query_value = WeightedValue(query_position);
  // assert(RNIsEqual(fabs(query_value), max_weighted_magnitude));
  return radius;
}

void R3Quadric::Draw(void) const {
  // For now, ???
  Outline();
}

void R3Quadric::Outline(void) const {
  // Initialize symmetry matrix
  static const R4Matrix mirror(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1);

  // Push symmetry matrix
  if (symmetry == 1) mirror.Push();

  // Outline support region
  support.Outline();

  // Pop symmetry matrix
  if (symmetry == 1) mirror.Pop();
}
