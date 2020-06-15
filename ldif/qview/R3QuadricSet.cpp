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
// Source file for 3D quadric set class

// Include files

namespace gaps {};
using namespace gaps;  // NOLINT(build/namespaces)

#include "R3Shapes/R3Shapes.h"

#include "R3Quadric.h"
#include "R3QuadricSet.h"

// Member functions

R3QuadricSet::R3QuadricSet(void)
    : quadrics(),
      bbox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX),
      flags(R3_QUADRIC_SET_PARTITION_OF_UNITY |
            R3_QUADRIC_SET_ALL_MATRICES_CONSTANT),
      occnet(NULL),
      isolevel(0),
      grid(NULL),
      mesh(NULL),
      res(0) {}

R3QuadricSet::R3QuadricSet(const R3Box &bbox)
    : quadrics(),
      bbox(bbox),
      flags(R3_QUADRIC_SET_PARTITION_OF_UNITY |
            R3_QUADRIC_SET_ALL_MATRICES_CONSTANT),
      occnet(NULL),
      isolevel(0),
      grid(NULL),
      mesh(NULL),
      res(0) {}

R3QuadricSet::R3QuadricSet(const R3QuadricSet &set)
    : quadrics(),
      bbox(set.bbox),
      flags(set.flags),
      occnet(NULL),
      isolevel(set.isolevel),
      grid(NULL),
      mesh(NULL),
      res(set.res) {
  // Copy quadrics
  for (int i = 0; i < set.NQuadrics(); i++) {
    const R3Quadric *q = set.Quadric(i);
    Insert(*q);
  }
}

R3QuadricSet::~R3QuadricSet(void) {
  // Empty
  Empty();
}

void R3QuadricSet::Insert(const R3Quadric &quadric) {
  // Add quadric
  quadrics.Insert(new R3Quadric(quadric));

  // Update bounding box
  bbox.Union(quadric.BBox());

  // Update flags
  if (!quadric.IsMatrixConstant()) {
    flags.Remove(R3_QUADRIC_SET_ALL_MATRICES_CONSTANT);
  }

  // Invalidate display stuff
  InvalidateGrid();
  InvalidateMesh();
}

void R3QuadricSet::Replace(int k, const R3Quadric &quadric) {
  // Delete previous quadric
  R3Quadric *previous_quadric = quadrics[k];

  // Replace quadric
  quadrics[k] = new R3Quadric(quadric);

  // Update bounding box
  bbox.Union(quadric.BBox());

  // Update flags
  if (!quadric.IsMatrixConstant()) {
    flags.Remove(R3_QUADRIC_SET_ALL_MATRICES_CONSTANT);
  }

  // Invalidate display stuff
  InvalidateGrid();
  InvalidateMesh();

  // Delete previous quadric
  delete previous_quadric;
}

void R3QuadricSet::Empty(void) {
  // Delete quadrics
  if (!quadrics.IsEmpty()) {
    for (int i = 0; i < quadrics.NEntries(); i++) delete quadrics[i];
    quadrics.Empty();
  }

  // Invalidate display stuff
  InvalidateGrid();
  InvalidateMesh();
  flags = 0;
}

void R3QuadricSet::Translate(const R3Vector &vector) {
  // Translate all quadrics
  for (int i = 0; i < quadrics.NEntries(); i++) {
    quadrics[i]->Translate(vector);
  }

  // Translate bounding box
  bbox.Translate(vector);

  // Update mesh
  if (mesh) {
    R3Affine transformation = R3identity_affine;
    transformation.Translate(vector);
    mesh->Transform(transformation);
  }

  // Invalidate grid
  InvalidateGrid();
}

void R3QuadricSet::Transform(const R3Transformation &transformation) {
  // Transform all quadrics
  for (int i = 0; i < quadrics.NEntries(); i++) {
    quadrics[i]->Transform(transformation);
  }

  // Transform bounding box
  // Note: since bbox is axial, grows box :(
  bbox.Transform(transformation);

  // Update mesh
  if (mesh) mesh->Transform(transformation);

  // Invalidate grid
  InvalidateGrid();
}

void R3QuadricSet::SetEvaluationMethod(RNFlags evaluation_method) {
  // Set evaluation method
  this->flags.Remove(R3_QUADRIC_SET_PARTITION_OF_UNITY);
  this->flags.Remove(R3_QUADRIC_SET_WEIGHTED_VALUE);
  this->flags.Remove(R3_QUADRIC_SET_WEIGHT);
  this->flags.Add(evaluation_method);

  // Invalidate grid
  InvalidateGrid();
}

void R3QuadricSet::SetResolution(int res) {
  // Set target grid resolution
  this->res = res;

  // Invalidate grid
  InvalidateGrid();
}

RNScalar R3QuadricSet::Value(const R3Point &position) const {
  // Return value
  if (flags[R3_QUADRIC_SET_PARTITION_OF_UNITY])
    return PartitionOfUnityValue(position);
  else if (flags[R3_QUADRIC_SET_WEIGHTED_VALUE])
    return WeightedValue(position);
  else if (flags[R3_QUADRIC_SET_WEIGHT])
    return Weight(position);
  else
    return 0;
}

RNScalar R3QuadricSet::PartitionOfUnityValue(const R3Point &position) const {
  // Return weighted sum divided by weight
  RNScalar weight = Weight(position);
  if (RNIsZero(weight)) return 0;
  RNScalar weighted_sum = WeightedValue(position);
  return weighted_sum / weight;
}

RNScalar R3QuadricSet::WeightedValue(const R3Point &position) const {
  // Sum weighted value of all quadrics
  RNScalar sum = 0;
  // TODO(kgenova) We should have a batch-eval function
  // that sets the embedding before evaluating the set of points.
  for (int i = 0; i < quadrics.NEntries(); i++) {
    // this->occnet->SetShapeEmbedding
    sum += quadrics[i]->WeightedValue(position);
  }
  return sum;
}

RNScalar R3QuadricSet::Weight(const R3Point &position) const {
  // Sum weight of all quadrics
  RNScalar sum = 0;
  for (int i = 0; i < quadrics.NEntries(); i++) {
    sum += quadrics[i]->Weight(position);
  }
  return sum;
}

int R3QuadricSet::ReadFile(const char *filename) {
  // Read file of appropriate type
  return ReadAsciiFile(filename);
}

int R3QuadricSet::WriteFile(const char *filename) const {
  // Write file of appropriate type
  return WriteAsciiFile(filename);
}

int R3QuadricSet::ReadAsciiFile(const char *filename) {
  // Open file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    RNFail("Unable to open quadric set file %s\n", filename);
    return 0;
  }

  // Read magic string
  char magic[128];
  if (fscanf(fp, "%s", magic) != (unsigned int)1) {
    RNFail("Error reading %s\n", filename);
    return 0;
  }

  // Check magic string
  if (strcmp(magic, "SIF")) {
    RNFail("Unrecognized key %s in first line of %s\n", magic, filename);
    return 0;
  }

  // Read header
  int nquadrics, version, nfeatures;
  if (fscanf(fp, "%d%d%d", &nquadrics, &version, &nfeatures) !=
      (unsigned int)3) {
    RNFail("Error reading header in %s\n", filename);
    return 0;
  }

  // Allocate buffer for features
  RNScalar *features = new RNScalar[nfeatures];
  if (!features) {
    RNFail("Unable to allocate features for %s\n", filename);
    return 0;
  }

  // Read lines from file
  for (int i = 0; i < nquadrics; i++) {
    // Read constant
    double m33;
    if (fscanf(fp, "%lf", &m33) != (unsigned int)1) {
      RNFail("Error reading constant of quadric %d in %s\n",
             quadrics.NEntries(), filename);
      break;
    }

    // Read center
    double c[3];
    if (fscanf(fp, "%lf%lf%lf", &c[0], &c[1], &c[2]) != (unsigned int)3) {
      RNFail("Error reading center of quadric %d in %s\n", quadrics.NEntries(),
             filename);
      return 0;
    }

    // Read radii
    double r[3];
    if (fscanf(fp, "%lf%lf%lf", &r[0], &r[1], &r[2]) != (unsigned int)3) {
      RNFail("Error reading radii of quadric %d in %s\n", quadrics.NEntries(),
             filename);
      return 0;
    }

    // Read rotation
    double a[3];
    if (fscanf(fp, "%lf%lf%lf", &a[0], &a[1], &a[2]) != (unsigned int)3) {
      RNFail("Error reading rotation of quadric %d in %s\n",
             quadrics.NEntries(), filename);
      return 0;
    }

    // Read symmetry
    int symmetry;
    if (fscanf(fp, "%d", &symmetry) != (unsigned int)1) {
      RNFail("Error reading symmetry of quadric %d in %s\n",
             quadrics.NEntries(), filename);
      return 0;
    }

    // Read features
    for (int j = 0; j < nfeatures; j++) {
      if (fscanf(fp, "%lf", &features[j]) != (unsigned int)1) {
        RNFail("Error reading features of quadric %d in %s\n",
               quadrics.NEntries(), filename);
        return 0;
      }
    }

    // Create quadric
    R4Matrix matrix(R4null_matrix);
    matrix[3][3] = m33;
    R3Point center(c);
    R3Vector radii(r);
    R3Triad triad(R3xyz_triad);
    triad.Rotate(RN_X, a[0]);
    triad.Rotate(RN_Y, a[1]);
    triad.Rotate(RN_Z, a[2]);
    R3CoordSystem cs(center, triad);
    R3Ellipsoid support(cs, radii);

    // Create quadric
    R3Quadric quadric(matrix, support, 0, features, nfeatures);
    Insert(quadric);

    // Create symmetric quadric
    if (symmetry == 1) {
      R3Quadric quadric(matrix, support, symmetry, features, nfeatures);
      Insert(quadric);
    }
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}

int R3QuadricSet::WriteAsciiFile(const char *filename) const {
  // Open file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    RNFail("Unable to open quadric set file %s\n", filename);
    return 0;
  }

  // Write magic string
  fprintf(fp, "SIF\n");

  // Write header
  int nfeatures = (NQuadrics() > 0) ? Quadric(0)->NFeatures() : 0;
  fprintf(fp, "%d %d %d\n", NQuadrics(), 0, nfeatures);

  // Write quadrics
  for (int i = 0; i < NQuadrics(); i++) {
    const R3Quadric *quadric = Quadric(i);
    assert(quadric->NFeatures() == nfeatures);

    // Write constant
    fprintf(fp, "%g ", quadric->Matrix()[3][3]);

    // Write center
    for (int j = 0; j < 3; j++) {
      fprintf(fp, "%g ", quadric->Center()[j]);
    }

    // Write radii
    for (int j = 0; j < 3; j++) {
      fprintf(fp, "%g ", quadric->Radius(j));
    }

    // Write symmetry
    fprintf(fp, "%d ", quadric->Symmetry());

    // Write features
    for (int j = 0; j < nfeatures; j++) {
      fprintf(fp, "%g ", quadric->Feature(j));
    }

    // Write newline
    fprintf(fp, "\n");
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}

int R3QuadricSet::ReadOldAsciiFile(const char *filename) {
  // Open file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    RNFail("Unable to open quadric set file %s\n", filename);
    return 0;
  }

  // Read lines from file
  while (TRUE) {
    // Read matrix
    double m[16];
    if (fscanf(fp, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf", &m[0],
               &m[1], &m[2], &m[3], &m[4], &m[5], &m[6], &m[7], &m[8], &m[9],
               &m[10], &m[11], &m[12], &m[13], &m[14],
               &m[15]) != (unsigned int)16) {
      break;
    }

    // Read center
    double c[3];
    if (fscanf(fp, "%lf%lf%lf", &c[0], &c[1], &c[2]) != (unsigned int)3) {
      RNFail("Error reading center of quadric %d in %s\n", quadrics.NEntries(),
             filename);
      return 0;
    }

    // Read radii
    double r[3];
    if (fscanf(fp, "%lf%lf%lf", &r[0], &r[1], &r[2]) != (unsigned int)3) {
      RNFail("Error reading radii of quadric %d in %s\n", quadrics.NEntries(),
             filename);
      return 0;
    }

    // Read rotation
    double a[3];
    if (fscanf(fp, "%lf%lf%lf", &a[0], &a[1], &a[2]) != (unsigned int)3) {
      RNFail("Error reading rotation of quadric %d in %s\n",
             quadrics.NEntries(), filename);
      return 0;
    }

    // Create quadric
    R4Matrix matrix(m);
    R3Point center(c);
    R3Vector radii(r);
    R3Triad triad(R3xyz_triad);
    triad.Rotate(RN_X, a[0]);
    triad.Rotate(RN_Y, a[1]);
    triad.Rotate(RN_Z, a[2]);
    R3CoordSystem cs(center, triad);
    R3Ellipsoid support(cs, radii);
    R3Quadric quadric(matrix, support);

    // Insert quadric
    Insert(quadric);
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}

int R3QuadricSet::WriteOldAsciiFile(const char *filename) const {
  // Open file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    RNFail("Unable to open quadric set file %s\n", filename);
    return 0;
  }

  // Write quadrics
  for (int i = 0; i < NQuadrics(); i++) {
    const R3Quadric *quadric = Quadric(i);

    // Write matrix
    const R4Matrix &m = quadric->Matrix();
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        fprintf(fp, "%g ", m[j][k]);
      }
    }

    // Write center
    for (int j = 0; j < 3; j++) {
      fprintf(fp, "%g ", quadric->Center()[j]);
    }

    // Write radii
    for (int j = 0; j < 3; j++) {
      fprintf(fp, "%g ", quadric->Radius(j));
    }

    // Write newline
    fprintf(fp, "\n");
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}

int R3QuadricSet::WriteIsosurfaceFile(const char *filename,
                                      RNScalar isolevel) const {
  // Update mesh
  ((R3QuadricSet *)this)->UpdateMesh(isolevel);
  if (!mesh) return 0;

  // Write mesh
  return mesh->WriteFile(filename);
}

void R3QuadricSet::Draw(RNScalar isolevel) const {
  // Draw mesh
  DrawIsoSurface(isolevel);
}

void R3QuadricSet::DrawIsoSurface(RNScalar isolevel) const {
  // Update mesh
  ((R3QuadricSet *)this)->UpdateMesh(isolevel);
  if (!mesh) return;

  // Draw wireframe
  mesh->DrawFaces();
}

void R3QuadricSet::DrawQuadricSupports(void) const {
  // Draw quadric supports
  for (int i = 0; i < NQuadrics(); i++) {
    const R3Quadric *quadric = Quadric(i);
    quadric->Support().Outline();
  }
}

void R3QuadricSet::UpdateGrid(void) {
  // Check stuff
  if (grid) return;
  if (bbox.IsEmpty()) return;

  // Update resolution
  if (res == 0) res = 256;

  // Compute grid spacing
  RNLength grid_spacing = bbox.LongestAxisLength() / res;
  if (grid_spacing == 0) return;

  // Allocate grid
  grid = new R3Grid(bbox, grid_spacing, 5, 512);
  if (!grid) {
    RNFail("Unable to allocate grid\n");
    return;
  }

  // Compute grid
  if (flags[R3_QUADRIC_SET_ALL_MATRICES_CONSTANT]) {
    // Splat values for each quadric
    RNScalar cutoff_magnitude = 1E-3;
    for (int i = 0; i < NQuadrics(); i++) {
      const R3Quadric *quadric = Quadric(i);
      RNScalar constant_value = quadric->UnweightedValue(R3zero_point);
      if (fabs(constant_value) <= cutoff_magnitude) continue;
      RNScalar ellipsoid_radius = quadric->Radius(RN_X);
      RNScalar cutoff_radius = quadric->Radius(RN_X, cutoff_magnitude);
      RNScalar inflation = (RNIsPositive(ellipsoid_radius))
                               ? cutoff_radius / ellipsoid_radius
                               : 1;
      R3Box quadric_bbox = quadric->BBox();
      quadric_bbox.Inflate(inflation);
      R3Point p1 = grid->GridPosition(quadric_bbox.Min());
      R3Point p2 = grid->GridPosition(quadric_bbox.Max());
      int ix1 = p1.X(), ix2 = p2.X() + 1;
      int iy1 = p1.Y(), iy2 = p2.Y() + 1;
      int iz1 = p1.Z(), iz2 = p2.Z() + 1;
      if (ix1 < 0) ix1 = 0;
      if (ix2 < 0) ix2 = 0;
      if (iy1 < 0) iy1 = 0;
      if (iy2 < 0) iy2 = 0;
      if (iz1 < 0) iz1 = 0;
      if (iz2 < 0) iz2 = 0;
      if (ix1 >= grid->XResolution()) ix1 = grid->XResolution() - 1;
      if (ix2 >= grid->XResolution()) ix2 = grid->XResolution() - 1;
      if (iy1 >= grid->YResolution()) iy1 = grid->YResolution() - 1;
      if (iy2 >= grid->YResolution()) iy2 = grid->YResolution() - 1;
      if (iz1 >= grid->ZResolution()) iz1 = grid->ZResolution() - 1;
      if (iz2 >= grid->ZResolution()) iz2 = grid->ZResolution() - 1;
      for (int iz = iz1; iz <= iz2; iz++) {
        for (int iy = iy1; iy <= iy2; iy++) {
          for (int ix = ix1; ix <= ix2; ix++) {
            R3Point grid_position(ix, iy, iz);
            R3Point world_position = grid->WorldPosition(grid_position);
            RNScalar value = quadric->WeightedValue(world_position);
            grid->AddGridValue(ix, iy, iz, value);
          }
        }
      }
    }
  } else {
    // Evaluate at every grid cell
    for (int iz = 0; iz < grid->ZResolution(); iz++) {
      for (int iy = 0; iy < grid->YResolution(); iy++) {
        for (int ix = 0; ix < grid->XResolution(); ix++) {
          R3Point grid_position(ix, iy, iz);
          R3Point world_position = grid->WorldPosition(grid_position);
          RNScalar value = Value(world_position);
          grid->SetGridValue(ix, iy, iz, value);
        }
      }
    }
  }

  // printf("HERE %g %g\n", grid->Minimum(), grid->Maximum());
  // grid->WriteFile("quadrics.grd");

  // Invalidate mesh
  InvalidateMesh();
}

void R3QuadricSet::UpdateMesh(RNScalar isolevel) {
  // Check stuff
  if (isolevel != this->isolevel) InvalidateMesh();
  if (mesh) return;
  UpdateGrid();
  if (!grid) return;

  // Allocate mesh
  mesh = new R3Mesh();
  if (!mesh) {
    RNFail("Unable to allocate mesh\n");
    return;
  }

  // Generate mesh
  grid->GenerateIsoSurface(isolevel, mesh);

  // Remember isolevel
  this->isolevel = isolevel;
}

void R3QuadricSet::InvalidateGrid(void) {
  // Delete grid
  if (grid) {
    delete grid;
    grid = NULL;
  }
}

void R3QuadricSet::InvalidateMesh(void) {
  // Delete mesh
  if (mesh) {
    delete mesh;
    mesh = NULL;
  }

  // Set isolevel to a weird value
  isolevel = 124578;
}
