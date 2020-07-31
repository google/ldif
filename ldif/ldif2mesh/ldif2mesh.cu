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
#include <assert.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <math.h>

#include <chrono>  // NOLINT(build/c++11)
#include <fstream>
#include <iostream>
#include <vector>

// Final value is in first element of array.
__device__ __host__ void ReduceSharedMem(float* in_out, int size) {
#ifdef __CUDA_ARCH__
  // This function is not safe for non power-of-2 sizes
  for (int s = size / 2; s >= 1; s >>= 1) {
    if (threadIdx.x < s) {
      in_out[threadIdx.x] += in_out[threadIdx.x + s];
    }
    __syncthreads();
  }
#else
  for (int i = 1; i < size; ++i) {
    in_out[0] += in_out[i];
  }
#endif
}

__device__ __host__ void SetRBFsBig(const float* rbfs, int size, float thresh,
                                    char* shared_memory) {
  bool* buffer = reinterpret_cast<bool*>(shared_memory);
#ifdef __CUDA_ARCH__
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    buffer[i] = rbfs[i] > thresh;
  }
  __syncthreads();
#else
  for (int i = 0; i < size; ++i) {
    buffer[i] = rbfs[i] > thresh;
  }
#endif
}

__device__ bool Reduce32(volatile bool* in_out) {
  // we could do + 32 here as well (32 threads-per-warp), but suspect the
  // kernel will often be called with exactly 32 threads:
  in_out[threadIdx.x] |= in_out[threadIdx.x + 16];
  in_out[threadIdx.x] |= in_out[threadIdx.x + 8];
  in_out[threadIdx.x] |= in_out[threadIdx.x + 4];
  in_out[threadIdx.x] |= in_out[threadIdx.x + 2];
  in_out[threadIdx.x] |= in_out[threadIdx.x + 1];
  return in_out[0];
}

// Final value is in first element of array.
__device__ __host__ bool ReduceAnySharedMem(bool* in_out, int size) {
#ifdef __CUDA_ARCH__
  // This function is not safe for non power-of-2 sizes
  if (size > 32) {
    for (int s = size / 2; s >= 32; s >>= 1) {
      if (threadIdx.x < s) {
        in_out[threadIdx.x] |= in_out[threadIdx.x + s];
      }
      __syncthreads();
    }
  }
  Reduce32(in_out);
  return in_out[0];
#else
  for (int i = 0; i < size; ++i) {
    if (in_out[i]) {
      return true;
    }
  }
  return false;
#endif
}

class FCLayer {
 public:
  float* weights;

  int input_size;
  int output_size;

  __device__ __host__ void Apply(const float* input, float* output) const {
#ifdef __CUDA_ARCH__
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
#else
    for (int i = 0; i < output_size; ++i) {
#endif
      float o = this->weights[input_size * output_size + i];  // bias
      for (int j = 0; j < input_size; ++j) {
        float ival = input[j];
        float wval = this->weights[j * output_size + i];
        o += ival * wval;
      }
      output[i] = o;
    }
#ifdef __CUDA_ARCH__
    __syncthreads();
#endif
  }

  __device__ __host__ float ApplyScalarOutSharedInPlace(float* input) const {
#ifdef __CUDA_ARCH__
    for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
      input[i] = input[i] * this->weights[i];
    }
    ReduceSharedMem(input, input_size);
    float output = input[0] + this->weights[input_size];
    return output;
#else
    float o = this->weights[input_size];  // bias
    for (int i = 0; i < input_size; ++i) {
      o += input[i] * this->weights[i];
    }
    return o;
#endif
  }

  __device__ __host__ void ApplyWithReluPreactivation(const float* input,
                                                      float* output) const {
#ifdef __CUDA_ARCH__
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
#else
    for (int i = 0; i < output_size; ++i) {
#endif
      float o = this->weights[input_size * output_size + i];
      for (int j = 0; j < input_size; ++j) {
        float ival = input[j] > 0 ? input[j] : 0.0f;
        float wval = this->weights[j * output_size + i];
        o += ival * wval;
      }
      output[i] = o;
    }
#ifdef __CUDA_ARCH__
    __syncthreads();
#endif
  }

  __device__ __host__ void ApplyWithSkip(const float* input,
                                         float* output) const {
#ifdef __CUDA_ARCH__
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
#else
    for (int i = 0; i < output_size; ++i) {
#endif
      float o = output[i] + this->weights[input_size * output_size + i];
      for (int j = 0; j < input_size; ++j) {
        float ival = input[j];
        float wval = this->weights[j * output_size + i];
        o += ival * wval;
      }
      output[i] = o;
    }
#ifdef __CUDA_ARCH__
    __syncthreads();
#endif
  }
  // Memory specification:
  // Read functions don't do their own mallocs. Instead, they assume the
  // pointers are already valid. then set their properties. There is a separate
  // class method "AssignMemory()". This function takes in a pointer, and makes
  // its values [weights, biases, e.g.] point to that. It also creates offsets
  // and calls AssignMemory for any member objects. There is a separate class
  // method "DynamicSize()". This function figures out the dynamic size of
  // itself, returns that size + the dynamic size of all its member objects.
  // This is used by a caller to figure out how big the initial assigment should
  // be. There is finally a class method BasePointer() which returns the pointer
  // initially given to AssignMemory(). This applies to all classes, including
  // LDIF. So the LDIF reader needs to change; Making an LDIF is now 1) allocate
  // the memory in a shared device-host space. 2) Give the LDIF the pointer. 3)
  // Read the ldif/occnet files. Now there is an LDIF that should work just the
  // same as the current one,
  //   except with a single cudaMallocManaged/delete.
  // Then, to switch over to shared memory:
  //   Make an LDIF in shared space. Start up the kernel, passing in shared
  //   memory. It needs to be sizeof(LDIF) + LDIF.DynamicSize() in size. The
  //   threads in the block (?this is the shared unit?) work together to copy
  //   the LDIF object to shared memory, and then synchronize. Both parts of the
  //   LDIF need to be copied at this stage by interpreting the ldif as a char*
  //   and sharing the copy workload, and doing the same for the pointer at
  //   ldif->Base() to the shared memory. I think the threads need to
  //   synchronize, one thread needs to set ldif->AssignMemory to the shared
  //   pointer, and then they need to synchronize again. Then the rest can
  //   happen, but in shared memory.
  //

  void Initialize(int row_count, int col_count) {
    this->input_size = row_count;
    this->output_size = col_count;
  }

  __device__ __host__ int DynamicSize() const {
    return (this->input_size + 1) * this->output_size * sizeof(float);
  }

  __device__ __host__ void AssignMemory(void* supplied) {
    this->weights = reinterpret_cast<float*>(supplied);
  }

  // Memory must be assigned prior to reading, or there will be undefined
  // behavior.
  void Read(std::ifstream* f) {
    f->read(reinterpret_cast<char*>(this->weights),
            sizeof(float) * (this->input_size + 1) * this->output_size);
  }
}
// Lint thinks this is unnecessary but NVCC requires it:
;  // NOLINT(whitespace/semicolon)

class CBNLayer {
 public:
  CBNLayer() { this->shape_set = false; }
  FCLayer beta_fc;
  FCLayer gamma_fc;
  float running_mean;
  float running_var;
  int element_count;
  int embedding_length;

  float* precomputed_betas;
  float* precomputed_gammas;

  // TODO(kgenova) This function creates undefined behavior if it is called
  // after the object has been moved to the GPU. Expose this in the API somehow,
  // so that an actual error is created instead of silently accessing invalid
  // memory.
  void ComputeActivations(const float* shape_embeddings) {
    // The input layer should have shape [#blobs, #elts]:
    assert(this->element_count == 32);
    assert(this->embedding_length == 32);
    // We don't currently malloc/free the activations as necessary:
    assert(!shape_set);
    for (int i = 0; i < this->element_count; ++i) {
      const float* shape_embedding_in =
          &shape_embeddings[this->element_count * i];
      this->beta_fc.Apply(shape_embedding_in,
                          &this->precomputed_betas[this->embedding_length * i]);
      this->gamma_fc.Apply(
          shape_embedding_in,
          &this->precomputed_gammas[this->embedding_length * i]);
    }
    this->shape_set = true;
  }

  __device__ __host__ void Apply(const float* sample_embedding, float* output,
                                 int element_index) {
    const float SQRT_EPS = 1e-5f;
    const float denom = sqrtf(this->running_var + SQRT_EPS);
    const float* beta =
        &this->precomputed_betas[this->embedding_length * element_index];
    const float* gamma =
        &this->precomputed_gammas[this->embedding_length * element_index];
    for (int i = 0; i < this->embedding_length; ++i) {
      output[i] = (sample_embedding[i] - this->running_mean) / denom;
      output[i] = gamma[i] * output[i] + beta[i];
    }
  }

  __device__ __host__ void ApplyInPlace(float* sample_embedding,
                                        int element_index) const {
    const float SQRT_EPS = 1e-5f;
    const float denom = sqrtf(this->running_var + SQRT_EPS);
    const float* beta =
        &this->precomputed_betas[this->embedding_length * element_index];
    const float* gamma =
        &this->precomputed_gammas[this->embedding_length * element_index];
#ifdef __CUDA_ARCH__
    for (int i = threadIdx.x; i < this->embedding_length; i += blockDim.x) {
#else
    for (int i = 0; i < this->embedding_length; ++i) {
#endif
      sample_embedding[i] = (sample_embedding[i] - this->running_mean) / denom;
      sample_embedding[i] = gamma[i] * sample_embedding[i] + beta[i];
    }
#ifdef __CUDA_ARCH__
    __syncthreads();
#endif
  }

  void Initialize(int row_count, int col_count, int element_count,
                  int embedding_length) {
    this->element_count = element_count;
    this->embedding_length = embedding_length;
    this->beta_fc.Initialize(embedding_length, embedding_length);
    this->gamma_fc.Initialize(embedding_length, embedding_length);
    // There is some CPU-only hidden memory that isn't part of what gets copied:
    // (this is a messy interface, but it keeps the DynamicSize() reporting what
    // actually needs to get copied, rather than the true dynamic size, which
    // makes it just barely fit in the 48kb of shared memory per block that CUDA
    // allows on 10-series cards.
    void* block_start_c =
        std::malloc(this->beta_fc.DynamicSize() + this->gamma_fc.DynamicSize());
    this->beta_fc.AssignMemory(block_start_c);
    char* gamma_start =
        reinterpret_cast<char*>(block_start_c) + this->gamma_fc.DynamicSize();
    this->gamma_fc.AssignMemory(reinterpret_cast<void*>(gamma_start));
  }

  __device__ __host__ int DynamicSize() const {
    // Precomputed CBN layers:
    int this_size =
        sizeof(float) * 2 * this->element_count * this->embedding_length;
    return this_size;
  }

  __device__ __host__ void AssignMemory(void* supplied) {
    this->precomputed_betas = reinterpret_cast<float*>(supplied);
    this->precomputed_gammas =
        this->precomputed_betas + this->element_count * this->embedding_length;
  }

  void Read(std::ifstream* f) {
    this->beta_fc.Read(f);
    this->gamma_fc.Read(f);
    f->read(reinterpret_cast<char*>(&this->running_mean), sizeof(float));
    f->read(reinterpret_cast<char*>(&this->running_var), sizeof(float));
  }

 private:
  bool shape_set;
}
// Lint thinks this is unnecessary but NVCC requires it:
;  // NOLINT(whitespace/semicolon)

__device__ __host__ void Relu(const float* input, float* output, int size) {
  for (int i = 0; i < size; ++i) {
    output[i] = input[i] > 0.0f ? input[i] : 0.0f;
  }
}

__device__ __host__ void ReluInPlace(float* input, int size) {
#ifdef __CUDA_ARCH__
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
#else
  for (int i = 0; i < size; ++i) {
#endif
    input[i] = input[i] > 0.0f ? input[i] : 0.0f;
  }
#ifdef __CUDA_ARCH__
  __syncthreads();
#endif
}

class OccNet {
 public:
  OccNet() { this->is_initialized = false; }
  void LoadFile(const std::string& path_to_occnet);
  void SetShape(float* implicits);
  __device__ __host__ float Eval(const float position[3], int element_index,
                                 char* shared_memory) const;
  bool IsInitialized() { return this->is_initialized; }
  void Initialize(int implicit_length, int element_count);
  __device__ __host__ int DynamicSize() const;
  __device__ __host__ void AssignMemory(void* supplied);

 private:
  int implicit_length;
  int element_count;
  FCLayer sample_resize_fc;
  FCLayer fc1;
  FCLayer fc2;
  FCLayer final_activation;
  CBNLayer cbn1;
  CBNLayer cbn2;
  CBNLayer cbnf;
  bool is_initialized;
  void* block_start;
};

void OccNet::SetShape(float* implicits) {
  this->cbn1.ComputeActivations(implicits);
  this->cbn2.ComputeActivations(implicits);
  this->cbnf.ComputeActivations(implicits);
}

__device__ __host__ float OccNet::Eval(const float position[3],
                                       int element_index,
                                       char* shared_memory) const {
  assert(this->is_initialized);
  float* buffer1 = reinterpret_cast<float*>(shared_memory);
  float* buffer2 = buffer1 + this->implicit_length;

  this->sample_resize_fc.Apply(position, buffer1);
  // post resize-fc now in buffer1
  this->cbn1.ApplyInPlace(buffer1, element_index);
  // post-relu1 now in buffer1
  this->fc1.ApplyWithReluPreactivation(buffer1,
                                       buffer2);    // post-fc1 now in buffer2
  this->cbn2.ApplyInPlace(buffer2, element_index);  // post-cbn2 in buffer2
  ReluInPlace(buffer2, this->implicit_length);      // post-relu2 in buffer2
  this->fc2.ApplyWithSkip(buffer2,
                          buffer1);  // post-skip (after post-fc2) in buffer1
  this->cbnf.ApplyInPlace(buffer1, element_index);  // post-cbnf in buffer1.
  float output = this->final_activation.ApplyScalarOutSharedInPlace(buffer1);
  return output;
}

void OccNet::Initialize(int implicit_length, int element_count) {
  this->implicit_length = implicit_length;
  this->element_count = element_count;
  this->sample_resize_fc.Initialize(3, implicit_length);
  this->cbn1.Initialize(implicit_length, implicit_length, element_count,
                        implicit_length);
  this->fc1.Initialize(implicit_length, implicit_length);
  this->cbn2.Initialize(implicit_length, implicit_length, element_count,
                        implicit_length);
  this->fc2.Initialize(implicit_length, implicit_length);
  this->cbnf.Initialize(implicit_length, implicit_length, element_count,
                        implicit_length);
  this->final_activation.Initialize(implicit_length, 1);
  this->is_initialized = true;
}

__device__ __host__ int OccNet::DynamicSize() const {
  int this_size = 0;
  this_size += this->sample_resize_fc.DynamicSize();
  this_size += this->cbn1.DynamicSize();
  this_size += this->fc1.DynamicSize();
  this_size += this->cbn2.DynamicSize();
  this_size += this->fc2.DynamicSize();
  this_size += this->cbnf.DynamicSize();
  this_size += this->final_activation.DynamicSize();
  return this_size;
}

__device__ __host__ void OccNet::AssignMemory(void* supplied) {
  char* base = reinterpret_cast<char*>(supplied);
  this->sample_resize_fc.AssignMemory(supplied);
  base += this->sample_resize_fc.DynamicSize();

  this->cbn1.AssignMemory(reinterpret_cast<void*>(base));
  base += this->cbn1.DynamicSize();

  this->fc1.AssignMemory(reinterpret_cast<void*>(base));
  base += this->fc1.DynamicSize();

  this->cbn2.AssignMemory(reinterpret_cast<void*>(base));
  base += this->cbn2.DynamicSize();

  this->fc2.AssignMemory(reinterpret_cast<void*>(base));
  base += this->fc2.DynamicSize();

  this->cbnf.AssignMemory(reinterpret_cast<void*>(base));
  base += this->cbnf.DynamicSize();

  this->final_activation.AssignMemory(reinterpret_cast<void*>(base));
}

void OccNet::LoadFile(const std::string& path_to_occnet) {
  assert(this->is_initialized);
  std::ifstream f(path_to_occnet, std::ios::in | std::ios::binary);
  int resnet_layer_count, implicit_length;
  f.read(reinterpret_cast<char*>(&resnet_layer_count), sizeof(int));
  f.read(reinterpret_cast<char*>(&implicit_length), sizeof(int));
  assert(this->implicit_length == implicit_length);
  if (this->implicit_length != implicit_length) {
    std::cout << "Error: OccNet file has dimensions imcompatible with the SIF "
                 "it has been initialized for."
              << std::endl;
  }
  assert(resnet_layer_count == 1);
  assert(implicit_length == 32);
  this->sample_resize_fc.Read(&f);
  this->cbn1.Read(&f);
  this->fc1.Read(&f);
  this->cbn2.Read(&f);
  this->fc2.Read(&f);
  this->cbnf.Read(&f);
  this->final_activation.Read(&f);
  f.close();
}

class LDIF {
 public:
  explicit LDIF(const std::string& path_to_ldif);
  void LoadFile(const std::string& path_to_ldif);
  void LoadOccNet(const std::string& path_to_occnet) {
    assert(this->len_implicits > 0);
    assert(this->num_blobs > 0);  // Should not load occnet before loading LDIF?
    this->occnet.LoadFile(path_to_occnet);
    this->occnet.SetShape(this->implicits);
  }
  bool HasImplicits() { return this->len_implicits > 0; }
  __device__ __host__ float ParEval(const float* position,
                                    char* shared_memory) const;
  __device__ __host__ void EvalRBFs(const float* position,
                                    char* shared_memory) const;
  __device__ __host__ float Eval(const float* position,
                                 char* shared_memory) const;
  __device__ __host__ float Eval(const float* position, int element_index,
                                 char* shared_memory) const;
  __device__ __host__ void World2Local(const float* position, int element_index,
                                       float* local_position) const;
  // LDIF doesn't have an initialize, because it is top-level and can just load
  // from a file. It does have a DynamicSize() and an AssignMemory(), so that it
  // can be copied to shared memory. This means that LoadFile needs to then call
  // and AssignMemory() itself, because the memory can't be allocated until it's
  // known how much to use by reading the file.
  __device__ __host__ int DynamicSize();
  __device__ __host__ void AssignMemory(void* supplied);

  //  private:
  float* constants;  // vector
  float* centers;    // column-major
  float* radii;      // row-major
  float* rotations;  // row-major
  int symmetry_count;
  float* implicits;  // row-major
  void* block_start;

  float* rotation_matrices;  // column-major
  int num_blobs;
  int len_implicits;

  OccNet occnet;

  void BuildRotation(int index);
  __device__ __host__ void BuildRollPitchYaw(const float rotation[3],
                                             float* output) const;
};

__device__ __host__ void Transpose3x3(const float* input, float* output) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      output[i * 3 + j] = input[j * 3 + i];
    }
  }
}

void MatMul3x3(const float* a, const float* b, float* output) {
  // ori = output row index
  // oci = ouotput column index
  // lici = left input column index
  // riri = right input row index
  // storage order is row-major
  for (int ori = 0; ori < 3; ++ori) {
    for (int oci = 0; oci < 3; ++oci) {
      output[ori * 3 + oci] = 0.0f;
      for (int i = 0; i < 3; ++i) {
        int lici = i;
        int riri = i;
        output[ori * 3 + oci] += a[ori * 3 + lici] * b[riri * 3 + oci];
      }
    }
  }
}

__device__ __host__ void MatVec3x3(const float* m, const float* v,
                                   float out[3]) {
  for (int ri = 0; ri < 3; ++ri) {
    out[ri] = 0.0f;
    for (int ci = 0; ci < 3; ++ci) {
      out[ri] += m[ri * 3 + ci] * v[ci];
    }
  }
}

__device__ __host__ void LDIF::World2Local(const float* position,
                                           int element_index,
                                           float* local_position) const {
  // Implicit RowVector to vector:
  float centered_point[3];
  centered_point[0] =
      position[0] - this->centers[this->num_blobs * 0 + element_index];
  centered_point[1] =
      position[1] - this->centers[this->num_blobs * 1 + element_index];
  centered_point[2] =
      position[2] - this->centers[this->num_blobs * 2 + element_index];
  float roll_pitch_yaw[9];
  this->BuildRollPitchYaw(&this->rotations[3 * element_index], roll_pitch_yaw);
  // TODO(kgenova) transpose of a rotation matrix is its inverse,
  // could be clever about row vectors
  float rpy_t[9];
  Transpose3x3(roll_pitch_yaw, rpy_t);
  float rotated_point[3];
  MatVec3x3(rpy_t, centered_point, rotated_point);
  float scale[3];
  scale[0] =
      (1.0f / (sqrtf(this->radii[element_index * 3 + 0] + 1e-8f) + 1e-8f));
  scale[1] =
      (1.0f / (sqrtf(this->radii[element_index * 3 + 1] + 1e-8f) + 1e-8f));
  scale[2] =
      (1.0f / (sqrtf(this->radii[element_index * 3 + 2] + 1e-8f) + 1e-8f));
  local_position[0] = scale[0] * rotated_point[0];
  local_position[1] = scale[1] * rotated_point[1];
  local_position[2] = scale[2] * rotated_point[2];
}

LDIF::LDIF(const std::string& path_to_ldif) { this->LoadFile(path_to_ldif); }

__device__ __host__ void LDIF::BuildRollPitchYaw(const float rotation[3],
                                                 float* output) const {
  float c[3];
  c[0] = cosf(rotation[0]);
  c[1] = cosf(rotation[1]);
  c[2] = cosf(rotation[2]);
  float s[3];
  s[0] = sinf(rotation[0]);
  s[1] = sinf(rotation[1]);
  s[2] = sinf(rotation[2]);
  output[0] = c[2] * c[1];
  output[1] = c[2] * s[1] * s[0] - s[2] * c[0];
  output[2] = c[2] * s[1] * c[0] + s[2] * s[0];
  output[3] = s[2] * c[1];
  output[4] = s[2] * s[1] * s[0] + c[2] * c[0];
  output[5] = s[2] * s[1] * c[0] - c[2] * s[0];
  output[6] = -s[1];
  output[7] = c[1] * s[0];
  output[8] = c[1] * c[0];
}

void LDIF::BuildRotation(int index) {
  float rpy[9];
  this->BuildRollPitchYaw(&this->rotations[3 * index], rpy);
  float rpy_t[9];
  Transpose3x3(rpy, rpy_t);
  float diag[9];
  diag[0] = 1.0 / (this->radii[3 * index + 0] + 1e-8f);
  diag[1] = 0.0;
  diag[2] = 0.0;
  diag[3] = 0.0;
  diag[4] = 1.0 / (this->radii[3 * index + 1] + 1e-8f);
  diag[5] = 0.0;
  diag[6] = 0.0;
  diag[7] = 0.0;
  diag[8] = 1.0 / (this->radii[3 * index + 2] + 1e-8f);
  float buffer[9];
  MatMul3x3(rpy, diag, buffer);
  float output_buffer[9];
  MatMul3x3(buffer, rpy_t, output_buffer);
  for (int i = 0; i < 9; ++i) {
    this->rotation_matrices[i * this->num_blobs + index] = output_buffer[i];
  }

  // Final output is rpy * diag * rpy_t:
}

__device__ __host__ void LDIF::EvalRBFs(const float* position,
                                        char* shared_memory) const {
  float* rbf_buffer = reinterpret_cast<float*>(shared_memory);
  int nb = this->num_blobs;
#ifdef __CUDA_ARCH__
  for (int i = threadIdx.x; i < nb; i += blockDim.x) {
#else
  for (int i = 0; i < nb; ++i) {
#endif
    const float* rm = this->rotation_matrices;
    float diff[3];
    diff[0] = position[0] - this->centers[0 * nb + i];
    diff[1] = position[1] - this->centers[1 * nb + i];
    diff[2] = position[2] - this->centers[2 * nb + i];
    float dist = diff[0] * (rm[(0 * 3 + 0) * nb + i] * diff[0] +
                            rm[(0 * 3 + 1) * nb + i] * diff[1] +
                            rm[(0 * 3 + 2) * nb + i] * diff[2]) +
                 diff[1] * (rm[(0 * 3 + 1) * nb + i] * diff[0] +
                            rm[(1 * 3 + 1) * nb + i] * diff[1] +
                            rm[(1 * 3 + 2) * nb + i] * diff[2]) +
                 diff[2] * (rm[(0 * 3 + 2) * nb + i] * diff[0] +
                            rm[(1 * 3 + 2) * nb + i] * diff[1] +
                            rm[(2 * 3 + 2) * nb + i] * diff[2]);
    rbf_buffer[i] = expf(-0.5f * dist);
  }
#ifdef __CUDA_ARCH__
  __syncthreads();
#endif
}

__device__ __host__ float LDIF::ParEval(const float* position,
                                        char* shared_memory) const {
  float sum = 0.0f;
  float flipped_position[3];
  flipped_position[0] = position[0];
  flipped_position[1] = position[1];
  flipped_position[2] = -position[2];
  // TODO(kgenova) This is an awkward dependency that should be addressed via a
  // member variable.
  char* rbf_buffer = shared_memory + sizeof(float) * this->len_implicits * 2;
  char* rbf_check_buffer = rbf_buffer + sizeof(float) * this->num_blobs;

  this->EvalRBFs(position, rbf_buffer);
  float* rbfs = reinterpret_cast<float*>(rbf_buffer);
  bool* rbfs_big = reinterpret_cast<bool*>(rbf_check_buffer);
  const float big = 0.0001f;  // 0.005f is fine when training on shapenet.
  SetRBFsBig(rbfs, this->num_blobs, big, rbf_check_buffer);
  if (ReduceAnySharedMem(reinterpret_cast<bool*>(rbf_check_buffer),
                         this->num_blobs)) {
    if (this->len_implicits == 0) {
      // TODO(kgenova) If we used a variant of a ParReduce here we could speed
      // up regular sifs.
      for (int i = 0; i < this->num_blobs; ++i) {
        sum += constants[i] *
               rbfs[i];  // Should also pull in constants in parallel.
      }
    } else {
      SetRBFsBig(rbfs, this->num_blobs, big, rbf_check_buffer);
      for (int i = 0; i < this->num_blobs; ++i) {
        if (rbfs_big[i]) {
          float local_position[3];
          this->World2Local(position, i, local_position);
          float occnet_val =
              this->occnet.Eval(local_position, i, shared_memory);
          sum += constants[i] * rbfs[i] * (1 + occnet_val);
        }
      }
    }
  }
  this->EvalRBFs(flipped_position, rbf_buffer);
  SetRBFsBig(rbfs, this->symmetry_count, big, rbf_check_buffer);
  if (ReduceAnySharedMem(rbfs_big, this->symmetry_count)) {
    if (this->len_implicits == 0) {
      // TODO(kgenova) If we used a variant of a ParReduce here we could speed
      // up regular sifs.
      for (int i = 0; i < this->symmetry_count; ++i) {
        sum += constants[i] *
               rbfs[i];  // Should also pull in constants in parallel.
      }
      return sum;
    } else {
      SetRBFsBig(rbfs, this->symmetry_count, big, rbf_check_buffer);
      for (int i = 0; i < this->symmetry_count; ++i) {
        if (rbfs_big[i]) {
          float local_position[3];
          this->World2Local(flipped_position, i, local_position);
          float occnet_val =
              this->occnet.Eval(local_position, i, shared_memory);
          sum += constants[i] * rbfs[i] * (1 + occnet_val);
        }
      }
    }
  }
  return sum;
}

__device__ __host__ int LDIF::DynamicSize() {
  int constant_size = this->num_blobs * sizeof(float);
  int center_size = this->num_blobs * 3 * sizeof(float);
  int radii_size = this->num_blobs * 3 * sizeof(float);
  int rotation_size = this->num_blobs * 3 * sizeof(float);
  int implicit_size = this->num_blobs * this->len_implicits * sizeof(float);
  int matrix_size = this->num_blobs * 3 * 3 * sizeof(float);
  return constant_size + center_size + radii_size + rotation_size +
         implicit_size + matrix_size + this->occnet.DynamicSize();
}

__device__ __host__ void LDIF::AssignMemory(void* supplied) {
  this->block_start = supplied;
  float* base = reinterpret_cast<float*>(supplied);

  this->constants = base;
  base += this->num_blobs;

  this->centers = base;
  base += this->num_blobs * 3;

  this->radii = base;
  base += this->num_blobs * 3;

  this->rotations = base;
  base += this->num_blobs * 3;

  this->implicits = base;
  base += this->num_blobs * this->len_implicits;

  this->rotation_matrices = base;
  base += this->num_blobs * 3 * 3;

  this->occnet.AssignMemory(reinterpret_cast<void*>(base));
}

void LDIF::LoadFile(const std::string& path_to_ldif) {
  std::ifstream f(path_to_ldif);
  assert(f.is_open());
  std::string line;
  getline(f, line);
  if (line != "SIF") {
    std::cout << "Expected header SIF but got " << line << std::endl;
  }
  int num_blobs, version_id, len_implicits;
  f >> num_blobs;
  f >> version_id;
  f >> len_implicits;
  assert(version_id == 0);
  assert(num_blobs >= 0);
  assert(len_implicits >= 0);
  this->num_blobs = num_blobs;
  this->len_implicits = len_implicits;
  this->occnet.Initialize(this->len_implicits, this->num_blobs);
  // We have now initialized the object, DynamicSize+AssignMemory are fair game.
  int size_to_malloc = this->DynamicSize();
  char* block_h = new char[size_to_malloc];
  this->AssignMemory(block_h);
  // Now that we have allocated and assigned memory, we can read the rest of the
  // file.

  bool found_asymmetric = false;
  for (int i = 0; i < num_blobs; ++i) {
    float constant;
    f >> constant;
    this->constants[i] = constant;
    f >> this->centers[0 * this->num_blobs + i] >>
        this->centers[1 * this->num_blobs + i] >>
        this->centers[2 * this->num_blobs + i];
    f >> this->radii[i * 3 + 0] >> this->radii[i * 3 + 1] >>
        this->radii[i * 3 + 2];
    this->radii[i * 3 + 0] = this->radii[i * 3 + 0] * this->radii[i * 3 + 0];
    this->radii[i * 3 + 1] = this->radii[i * 3 + 1] * this->radii[i * 3 + 1];
    this->radii[i * 3 + 2] = this->radii[i * 3 + 2] * this->radii[i * 3 + 2];
    f >> this->rotations[i * 3 + 0] >> this->rotations[i * 3 + 1] >>
        this->rotations[i * 3 + 2];
    int is_symmetric;
    f >> is_symmetric;
    assert(is_symmetric == 0 || is_symmetric == 1);
    assert(!(is_symmetric && found_asymmetric));
    if (!found_asymmetric && !is_symmetric) {
      this->symmetry_count = i;
      found_asymmetric = true;
    }
    for (int j = 0; j < len_implicits; ++j) {
      f >> this->implicits[i * len_implicits + j];
    }
  }
  for (int i = 0; i < num_blobs; ++i) {
    this->BuildRotation(i);
  }
}

static void WriteGrid(const std::string& path, float* grid, int resolution) {
  std::ofstream f(path, std::ios::out | std::ios::binary);
  assert(f.is_open());
  f.write(reinterpret_cast<char*>(&resolution), sizeof(int));
  f.write(reinterpret_cast<char*>(&resolution), sizeof(int));
  f.write(reinterpret_cast<char*>(&resolution), sizeof(int));

  // Write an identity transformation:
  float one = 1.0f;
  float zero = 0.0f;
  f.write(reinterpret_cast<char*>(&one), sizeof(float));
  f.write(reinterpret_cast<char*>(&zero), sizeof(float));
  f.write(reinterpret_cast<char*>(&zero), sizeof(float));
  f.write(reinterpret_cast<char*>(&zero), sizeof(float));
  f.write(reinterpret_cast<char*>(&zero), sizeof(float));
  f.write(reinterpret_cast<char*>(&one), sizeof(float));
  f.write(reinterpret_cast<char*>(&zero), sizeof(float));
  f.write(reinterpret_cast<char*>(&zero), sizeof(float));
  f.write(reinterpret_cast<char*>(&zero), sizeof(float));
  f.write(reinterpret_cast<char*>(&zero), sizeof(float));
  f.write(reinterpret_cast<char*>(&one), sizeof(float));
  f.write(reinterpret_cast<char*>(&zero), sizeof(float));
  f.write(reinterpret_cast<char*>(&zero), sizeof(float));
  f.write(reinterpret_cast<char*>(&zero), sizeof(float));
  f.write(reinterpret_cast<char*>(&zero), sizeof(float));
  f.write(reinterpret_cast<char*>(&one), sizeof(float));

  f.write(reinterpret_cast<char*>(grid),
          sizeof(float) * resolution * resolution * resolution);

  f.close();
}

__device__ void make_shared_static_ldif(int* shared_memory,
                                        const LDIF* global_ldif,
                                        const LDIF** shared_ldif) {
  const int* global_static_space = reinterpret_cast<const int*>(global_ldif);
  int static_size = sizeof(LDIF) / 4;
  for (int i = threadIdx.x; i < static_size; i += blockDim.x) {
    shared_memory[i] = global_static_space[i];
  }
  __syncthreads();
  *shared_ldif = reinterpret_cast<LDIF*>(shared_memory);
}

__device__ void make_shared_ldif(char* shared_memory, LDIF* global_ldif,
                                 LDIF** shared_ldif) {
  // Get an index that is unique in the block, because shared memory is unique
  // to the block:
  int block_serial_index = threadIdx.z * blockDim.y * blockDim.x +
                           threadIdx.y * blockDim.x + threadIdx.x;
  int serial_stride = blockDim.z * blockDim.y * blockDim.x;
  int static_size = sizeof(LDIF);
  char* global_static_space = reinterpret_cast<char*>(global_ldif);
  for (int i = block_serial_index; i < static_size; i += serial_stride) {
    shared_memory[i] = global_static_space[i];
  }
  __syncthreads();
  // Now that the LDIF is initialized, we can get a pointer to its dynamic
  // memory:
  int dynamic_size = global_ldif->DynamicSize();
  char* global_dynamic_space =
      reinterpret_cast<char*>(global_ldif->block_start);
  char* shared_dynamic_space = shared_memory + static_size;
  *shared_ldif = reinterpret_cast<LDIF*>(shared_memory);
  for (int i = block_serial_index; i < dynamic_size; i += serial_stride) {
    shared_dynamic_space[i] = global_dynamic_space[i];
  }
  __syncthreads();
  // Now the shared dynamic memory has been allocated; only one thread should do
  // the move:
  if (block_serial_index == 0)
    (*shared_ldif)->AssignMemory(reinterpret_cast<void*>(shared_dynamic_space));

  __syncthreads();
  // Now the shared_ldif object is fully initialized and contains no pointers to
  // global memory.
}

#define MAX_THREADS_PER_BLOCK 1000

__global__ void Eval(float* grid, const LDIF* ldif, char* ldif_dynamic_memory_g,
                     int resolution) {
  // float position[3] = {0.02, 0.008, -0.05};
  extern __shared__ char shared_memory_base[];

  const LDIF* shared_ldif;
  make_shared_static_ldif(reinterpret_cast<int*>(shared_memory_base), ldif,
                          &shared_ldif);
  char* scratch_shared_memory = shared_memory_base + sizeof(LDIF);

  float extent = 0.75;
  float total_size = 2 * extent;

  // LDIF* shared_ldif;
  // make_shared_ldif(s, ldif, &shared_ldif);

  // x index (within the block, threadIdx.x) says which of the 32 threads
  // working on the same grid element the thread is. blockIdx.x [0-2billion]
  // says which grid element to process.
  int width = blockIdx.x % resolution;
  int height = (blockIdx.x / resolution) % resolution;
  int depth = (blockIdx.x / (resolution * resolution)) % resolution;
  if (blockIdx.x >= resolution * resolution * resolution) {
    assert(false);
    return;
  }
  float x_max = resolution - 0.5;
  float x_interpolated = 0.5 + x_max * (static_cast<float>(width) / resolution);
  float x_fraction = x_interpolated / resolution;
  float y_max = resolution - 0.5;
  float y_interpolated =
      0.5 + y_max * (static_cast<float>(height) / resolution);
  float y_fraction = y_interpolated / resolution;
  float z_max = resolution - 0.5;
  float z_interpolated = 0.5 + z_max * (static_cast<float>(depth) / resolution);
  float z_fraction = z_interpolated / resolution;
  float position[3];
  position[0] = x_fraction * total_size - extent;
  position[1] = y_fraction * total_size - extent;
  position[2] = z_fraction * total_size - extent;
  float result = shared_ldif->ParEval(position, scratch_shared_memory);
  int grid_index =
      depth * resolution * resolution + height * resolution + width;
  assert(grid_index == blockIdx.x);
  if (threadIdx.x == 0) {
    grid[grid_index] = result;
  }
  __syncthreads();
}

#define GPU_CHECK_OK(code) { GPUCheckOk((code), __FILE__, __LINE__); }
inline void GPUCheckOk(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    std::cout << "GPUCheckOk Failure: " << cudaGetErrorString(code) << " "
        << file << " " << line << std::endl;
    exit(code);
  }
}

int main(int argc, char** argv) {
  auto start_t = std::chrono::high_resolution_clock::now();
  GPU_CHECK_OK(cudaFree(0));  // Get started warming cuda up asap.
  // Default to test paths for profiling:
  std::string input_ldif_path = "./test-ldif.txt";
  std::string occnet_path = "./extracted.occnet";
  std::string output_path = "./test-grd.grd";
  int resolution = 256;
  std::string usage_message =
      "Usage: ldif2mesh [input_ldif_path] [occnet_path] [output_path] "
      "[-resolution #].";
  if (argc > 1 && argc < 4) {
    std::cout << usage_message << std::endl;
    return 1;
  }
  if (argc > 1) {
    input_ldif_path = std::string(argv[1]);
    occnet_path = std::string(argv[2]);
    output_path = std::string(argv[3]);
  }
  if (argc > 4) {
    if (std::string(argv[4]) != "-resolution") {
      std::cout << usage_message << std::endl;
    }
    // Convert to string first to avoid atoi:
    resolution = std::stoi(std::string(argv[5]));
  }

  LDIF ldif(input_ldif_path);
  if (sizeof(LDIF) % 4) {
    std::cout << "Error! LDIF size invalid." << std::endl;
  }
  if (ldif.HasImplicits()) {
    ldif.LoadOccNet(occnet_path);
  }
  char* gpu_ldif_dynamic_memory;
  GPU_CHECK_OK(cudaMalloc(&gpu_ldif_dynamic_memory, ldif.DynamicSize()));
  GPU_CHECK_OK(cudaMemcpy(gpu_ldif_dynamic_memory, ldif.block_start,
                          ldif.DynamicSize(), cudaMemcpyHostToDevice));
  ldif.AssignMemory(reinterpret_cast<void*>(gpu_ldif_dynamic_memory));

  float* grid = 0;
  GPU_CHECK_OK(cudaMalloc(
      &grid, resolution * resolution * resolution * sizeof(float)));
  LDIF* gpu_ldif = 0;
  GPU_CHECK_OK(cudaMalloc(&gpu_ldif, sizeof(LDIF)));
  GPU_CHECK_OK(cudaMemcpy(gpu_ldif, &ldif, sizeof(LDIF),
                          cudaMemcpyHostToDevice));

  int fc_required_shared_size = sizeof(float) * ldif.len_implicits * 2;
  int sif_required_shared_size = sizeof(float) * ldif.num_blobs +
                                 sizeof(bool) * ldif.num_blobs + sizeof(LDIF);
  int shared_size = fc_required_shared_size + sif_required_shared_size;
  int N = resolution * resolution * resolution;
  int xres = 32;
  int num_blocks = N;
  auto kernel_start_t = std::chrono::high_resolution_clock::now();
  Eval<<<num_blocks, xres, shared_size>>>(grid, gpu_ldif,
                                          gpu_ldif_dynamic_memory, resolution);
  GPU_CHECK_OK(cudaGetLastError());
  GPU_CHECK_OK(cudaDeviceSynchronize());
  auto kernel_stop_t = std::chrono::high_resolution_clock::now();
  float* grid_h = new float[resolution * resolution * resolution];
  GPU_CHECK_OK(cudaMemcpy(grid_h, grid,
                          sizeof(float) * resolution * resolution * resolution,
                          cudaMemcpyDeviceToHost));
  // Dummy position for testing...
  // float position[3] = { 0.02, 0.008, -0.05 };
  WriteGrid(output_path, grid_h, resolution);
  auto stop_t = std::chrono::high_resolution_clock::now();
  std::chrono::microseconds runtime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop_t - start_t);
  std::chrono::microseconds kernel_runtime =
      std::chrono::duration_cast<std::chrono::microseconds>(kernel_stop_t -
                                                            kernel_start_t);
  float completed_in = runtime.count() * 1e-3f;
  float kernel_completed_in = kernel_runtime.count() * 1e-3f;
  std::cout << "Completed in " << std::to_string(completed_in)
            << " milliseconds (" << std::to_string(kernel_completed_in)
            << "ms kernel time)." << std::endl;
  GPU_CHECK_OK(cudaProfilerStop());
  return 0;
}
