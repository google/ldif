"""Minimal support for SIF evaluation in pytorch."""

def _load_v1_txt(path):
  """Parses a SIF V1 text file, returning numpy arrays.
  
  Args:
    path: string containing the path to the ASCII file.
    
  Returns:
    A tuple of 4 elements:
      constants: A numpy array of shape (element_count). The constant
        associated with each SIF element.
      centers: A numpy array of shape (element_count, 3). The centers of the
        SIF elements.
      radii: A numpy array of shape (element_count, 3). The axis-aligned
        radii of the gaussian falloffs.
      rotations: A numpy array of shape (element_count, 3). The euler-angle
        rotations of the SIF elements.
      symmetry_count: An integer. The number of elements which are left-right
        symmetric.
  """
  pass  



class Sif(object):
  """A SIF for loading from txts, packing into a tensor, and evaluation."""

  def __init__(self):
    pass

  @classmethod
  def from_file(cls, path):
    """Generates a SIF object from a txt file."""
    pass

  @classmethod
  def from_flat_tensor(self, tensor):
    """Generates a batch of SIFs from a batched tensor."""
    pass

  def to_flat_tensor(self):
    """Generates a flat tensor from the SIF. Can be batched with torch.stack."""
    pass
  
  def rbf_influence(self, samples):
    """Evaluates the influence of each RBF in the SIF at each sample.

    Args:
      samples: A tensor containing the samples, in the SIF global frame.
        Has shape (sample_count, 3) or (bs, sample_count_3).

    Returns:
      A tensor with shape (sample_count, effective_element_count) or
      (bs, sample_count, effective_element_count). The 'effective' element
      count may be higher than the element count, depending on the symmetry
      settings of the SIF. In the case were the SIF is at least partially
      symmetric, then some elements have multiple RBF weights- their main
      weight (given first) and the weight associated with the 'shadow'
      element(s) transformed by their symmetry matrix. See get_symmetry_map()
      for a mapping from original element indices to their symmetric 
      counterparts. Regardless of additional 'shadow' elements, the first 
      element_count RBF weights correspond to the 'real' elements with no
      symmetry transforms applied, in order.
    """
    pass

  def constants(self):
    """The constant parameters associated with the SIF.

    Returns:
      A tensor with shape (effective_element_count) or
      (bs, effective_element_count). See rbf_influence for an explanation
      of how to 'effective' samples.
    """
    pass

  def world2local(self):
    """The 4x4 transformation matrices associated with the SIF elements.

    Returns:
      A tensor of shape (effective_element_count, 4, 4) or 
      (bs, effective_element_count, 4, 4). See rbf_influence for an explanation
      of element_count vs effective_element_count.
    """
    pass

  def eval(self, samples):
    """Evaluates the SIF at the samples.

    Args:
      samples: A tensor of shape (sample_count, 3) or (bs, sample_count, 3).
        The locations to evaluate the SIF, in the SIF's world coordinate frame.

    Returns:
      A tensor of shape (sample_count) or (bs, sample_count). The value of the
        SIF at each sample point. Typically, values less than -0.07 are inside
        and values greater than -0.07 are outside.
    """
    # TODO(kgenova) A future version of the SIF txt file should contain the
    # isosurface used for inside/outside determination, so users don't have
    # to keep that information around.
    pass
      
