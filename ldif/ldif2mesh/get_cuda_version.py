# Lint as: python3
"""A shell utility to get the CUDA version."""

import subprocess as sp

def get_cuda_version():
  try:
    output = sp.check_output(['nvcc', '-V']).decode('utf-8')
    lines = output.split('\n')
    version_str = lines[-2].split(',')[1].split(' ')[-1]
    major_version = int(version_str.split('.')[0])
    minor_version = int(version_str.split('.')[1])
    return major_version, minor_version
  except Exception as e:
    raise ValueError(f'Failed to get cuda version with error: {e}')

if __name__ == '__main__':
  major, minor = get_cuda_version()
  print(major)
  print(minor)
