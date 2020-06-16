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
// Include file for 3D quadric set class
#ifndef R3QUADRIC_SET_H
#define R3QUADRIC_SET_H

#include "Occnet.h"

// Class definition

class R3QuadricSet {
 public:
  // Constructor/deconstructor
  R3QuadricSet(void);
  R3QuadricSet(const R3Box& bbox);
  R3QuadricSet(const R3QuadricSet& set);
  ~R3QuadricSet(void);

  // Access functions
  int NQuadrics(void) const;
  const R3Quadric* Quadric(int k) const;

  // Property functions
  const R3Box& BBox(void) const;

  // Insertion/removal functions
  void Insert(const R3Quadric& quadric);
  void Replace(int k, const R3Quadric& quadric);
  void Empty(void);

  // Manipulation functions
  void Translate(const R3Vector& vector);
  void Transform(const R3Transformation& transformation);
  void SetEvaluationMethod(RNFlags evaluation_method);

  // Implicit value functions
  RNScalar Value(const R3Point& position) const;
  RNScalar PartitionOfUnityValue(const R3Point& position) const;
  RNScalar WeightedValue(const R3Point& position) const;
  RNScalar Weight(const R3Point& position) const;

  // Read/write functions
  int ReadFile(const char* filename);
  int ReadAsciiFile(const char* filename);
  int WriteFile(const char* filename) const;
  int WriteAsciiFile(const char* filename) const;
  int WriteIsosurfaceFile(const char* filename, RNScalar isolevel = 0) const;

  // Display functions
  void Draw(RNScalar isolevel = 0) const;
  void DrawIsoSurface(RNScalar isolevel = 0) const;
  void DrawQuadricSupports(void) const;

 public:
  // Parameters to control display resolution
  void SetResolution(int res);

  // Update bookkeeping
  void UpdateGrid(void);
  void UpdateMesh(RNScalar isolevel = 0);
  void InvalidateGrid(void);
  void InvalidateMesh(void);

  // Read/write functions for old format
  int ReadOldAsciiFile(const char* filename);
  int WriteOldAsciiFile(const char* filename) const;

  void SetOccNet(OccNet* occnet);

 private:
  // Basic data
  RNArray<R3Quadric*> quadrics;
  R3Box bbox;
  RNFlags flags;
  OccNet* occnet;
  // std::vector<std::vector<float>> embeddings;

  // For display only
  RNScalar isolevel;
  R3Grid* grid;
  R3Mesh* mesh;
  int res;
};

// Constants

#define R3_QUADRIC_SET_ALL_MATRICES_CONSTANT 0x10
#define R3_QUADRIC_SET_PARTITION_OF_UNITY 0x01
#define R3_QUADRIC_SET_WEIGHTED_VALUE 0x02
#define R3_QUADRIC_SET_WEIGHT 0x04

// Inline functions

inline int R3QuadricSet::NQuadrics(void) const {
  // Return number of quadrics
  return quadrics.NEntries();
}

inline const R3Quadric* R3QuadricSet::Quadric(int k) const {
  // Return kth quadric
  return quadrics.Kth(k);
}

inline const R3Box& R3QuadricSet::BBox(void) const {
  // Return bounding box
  return bbox;
}

inline void R3QuadricSet::SetOccNet(OccNet* occnet) { this->occnet = occnet; }

#endif // R3QUADRIC_SET_H
