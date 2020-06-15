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
// Include file for 3D quadric class

// Class definition
#ifndef R3QUADRIC_H
#define R3QUADRIC_H

class R3Quadric {
 public:
  // Constructor/deconstructor
  R3Quadric(void);
  R3Quadric(const R3Quadric& quadric);
  R3Quadric(const R4Matrix& matrix, const R3Ellipsoid& support,
            int symmetry = 0, RNScalar* features = NULL, int nfeatures = 0);
  ~R3Quadric(void);

  // Property functions
  const R4Matrix& Matrix(void) const;
  const R3Ellipsoid& Support(void) const;
  const RNScalar ConstantTerm(void) const;
  const RNBoolean IsMatrixConstant(void) const;
  const R3CoordSystem& CoordSystem(void) const;
  const R3Point Center(void) const;
  const R3Vector Axis(int dim) const;
  const RNLength Radius(int dim) const;
  const RNLength Radius(int dim, RNScalar max_weighted_magnitude) const;
  const int Symmetry(void) const;
  const int NFeatures(void) const;
  const RNScalar Feature(int k) const;
  const R3Box BBox(void) const;

  // Manipulation functions
  void Empty(void);
  void SetMatrix(const R4Matrix& matrix);
  void SetSupport(const R3Ellipsoid& support);
  void SetSymmetry(int symmetry);
  void SetFeatures(const RNScalar* features, int nfeatures);
  void Translate(const R3Vector& vector);
  void Transform(const R3Transformation& transformation);

  // Implicit value functions
  RNScalar WeightedValue(const R3Point& position) const;
  RNScalar UnweightedValue(const R3Point& position) const;
  RNScalar Weight(const R3Point& position) const;

  // Display functions
  void Draw(void) const;
  void Outline(void) const;

 private:
  R4Matrix matrix;
  R3Ellipsoid support;
  RNScalar* features;
  int nfeatures;
  int symmetry;
  RNFlags flags;
};

// Constants

#define R3_QUADRIC_MATRIX_IS_CONSTANT 0x0001

// Inline functions

inline const R4Matrix& R3Quadric::Matrix(void) const {
  // Return matrix
  return matrix;
}

inline const R3Ellipsoid& R3Quadric::Support(void) const {
  // Return region of support
  return support;
}

inline const RNScalar R3Quadric::ConstantTerm(void) const {
  // Return constant term
  return matrix[3][3];
}

inline const RNBoolean R3Quadric::IsMatrixConstant(void) const {
  // Return whether matrix is a constant
  return flags[R3_QUADRIC_MATRIX_IS_CONSTANT];
}

inline const R3Point R3Quadric::Center(void) const {
  // Return center
  return support.Centroid();
}

inline const R3CoordSystem& R3Quadric::CoordSystem(void) const {
  // Return coordinate system
  return support.CoordSystem();
}

inline const R3Vector R3Quadric::Axis(int dim) const {
  // Return direction of dim axis
  return support.CoordSystem().Axes()[dim];
}

inline const RNLength R3Quadric::Radius(int dim) const {
  // Return radius of support along dim axis
  return support.Radii()[dim];
}

inline const int R3Quadric::Symmetry(void) const {
  // Return symmetry type
  return symmetry;
}

inline const int R3Quadric::NFeatures(void) const {
  // Return number of latent features
  return nfeatures;
}

inline const RNScalar R3Quadric::Feature(int k) const {
  // Return kth feature
  assert((k >= 0) && (k < nfeatures));
  return features[k];
}

inline const R3Box R3Quadric::BBox(void) const {
  // Return bounding box
  return support.BBox();
}

#endif // R3QUADRIC_H
