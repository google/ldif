# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Basic Definitions for defining and configuring the utility library."""

import abc
import glob
import os
import shutil

ENVIRONMENT = 'EXTERNAL'


class FileSystem(abc.ABC):
  """An abstract base class representing an interface to a filesystem."""

  @abc.abstractmethod
  def open(self, filename, mode):
    pass

  @abc.abstractmethod
  def mkdir(self, path, exist_ok=False):
    """Makes a directory.

    Args:
      path: String. The path to the directory to make.
      exist_ok: Boolean. If True, errors that the file already exists are
        suppressed. Otherwise, the function raises an exception if the directory
        already exists.
    """
    pass

  @abc.abstractmethod
  def makedirs(self, path, exist_ok=False):
    """Makes a directory tree recursively.

    Args:
      path: String. The path to the directory to make.
      exist_ok: Boolean. If True, errors that the file already exists are
        suppressed. Otherwise, the function raises an exception if the directory
        already exists.
    """
    pass

  @abc.abstractmethod
  def glob(self, path):
    pass

  @abc.abstractmethod
  def exists(self, path):
    pass

  @abc.abstractmethod
  def cp(self, from_path, to_path):
    """Copies a regular file (not a directory) to a new location.

    If a file already exists at the destination, or the source does not exist
    or is a directory, then behavior is unspecified.

    Args:
      from_path: String. The path to the source file.
      to_path: String. The path to the destination file.
    """
    # TODO(kgenova) This behavior should be better specified.
    pass

  def rm(self, path):
    """Removes a regular file (not a directory).

    If the file does not exist, permissions are insufficient, or the path
    points to a directory, then behavior is unspecified.

    Args:
      path: String. The path to the file to be removed.
    """
    pass


class StandardFileSystem(FileSystem):
  """A FileSystem that uses the standard os and built-in modules."""

  def mkdir(self, path, exist_ok=False):
    try:
      os.mkdir(path)
    except FileExistsError as e:
      if exist_ok:
        return
      raise FileExistsError('Passing through mkdir() error.') from e

  def makedirs(self, path, exist_ok=False):
    return os.makedirs(path, exist_ok=exist_ok)

  def open(self, *args):
    return open(*args)

  def glob(self, *args):
    return glob.glob(*args)

  def exists(self, *args):
    return os.path.exists(*args)

  def cp(self, *args):
    return shutil.copyfile(*args)

  def rm(self, *args):
    return os.remove(*args)


class Log(abc.ABC):
  """An abstract class representing a log for messages."""

  @abc.abstractmethod
  def log(self, msg, level='info'):
    """Logs a message to the underlying log."""
    pass

  @property
  def levels(self):
    return ['verbose', 'info', 'warning', 'error']

  def level_index(self, level):
    level = level.lower()
    if level not in self.levels:
      raise ValueError(f'Unrecognized logging level: {level}')
    i = 0
    for i in range(len(self.levels)):
      if self.levels[i] == level:
        return i
    assert False  # Should be unreachable


class SimpleLog(Log):
  """A log that just prints with a level indicator."""

  def __init__(self):
    super(SimpleLog, self).__init__()
    self.visible_levels = self.levels

  def log(self, msg, level='info'):
    if level.lower() not in self.levels:
      raise ValueError(f'Invalid logging level: {level}')
    if level.lower() not in self.visible_levels:
      return  # Too low level to display
    print(f'{level.upper()}: {msg}')

  def verbose(self, msg):
    self.log(msg, level='verbose')

  def info(self, msg):
    self.log(msg, level='info')

  def warning(self, msg):
    self.log(msg, level='warning')

  def error(self, msg):
    self.log(msg, level='error')

  def set_level(self, level):
    index = self.level_index(level)
    self.visible_levels = self.levels[index:]
    self.verbose(f'Logging level changed to {level}')


if ENVIRONMENT == 'GOOGLE':
  raise ValueError('Google file-system and logging no longer supported.')
elif ENVIRONMENT == 'EXTERNAL':
  FS = StandardFileSystem()
  LOG = SimpleLog()
else:
  raise ValueError(f'Unrecognized library mode: {ENVIRONMENT}.')
