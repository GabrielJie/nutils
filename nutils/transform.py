# Copyright (c) 2014 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
The transform module.
"""

from typing import Tuple, Dict
from . import cache, numeric, util, types, evaluable
from .evaluable import Evaluable, Array
import numpy, collections, itertools, functools, operator
_ = numpy.newaxis

TransformChain = Tuple['TransformItem']

## TRANSFORM CHAIN OPERATIONS

def apply(chain, points):
  # NOTE: we explicitly do not lru_cache apply, as doing so would create a
  # cyclic reference when chain is empty or contains only Identity transforms.
  # Instead we rely on the caching of individual transform items.
  for trans in reversed(chain):
    points = trans.apply(points)
  return points

def canonical(chain):
  # keep at lowest ndims possible; this is the required form for bisection
  n = len(chain)
  if n < 2:
    return tuple(chain)
  items = list(chain)
  i = 0
  while items[i].fromdims > items[n-1].fromdims:
    swapped = items[i+1].swapdown(items[i])
    if swapped:
      items[i:i+2] = swapped
      i -= i > 0
    else:
      i += 1
  return tuple(items)

def uppermost(chain):
  # bring to highest ndims possible
  n = len(chain)
  if n < 2:
    return tuple(chain)
  items = list(chain)
  i = n
  while items[i-1].todims < items[0].todims:
    swapped = items[i-2].swapup(items[i-1])
    if swapped:
      items[i-2:i] = swapped
      i += i < n
    else:
      i -= 1
  return tuple(items)

def promote(chain, ndims):
  # swap transformations such that ndims is reached as soon as possible, and
  # then maintained as long as possible (i.e. proceeds as canonical).
  for i, item in enumerate(chain): # NOTE possible efficiency gain using bisection
    if item.fromdims == ndims:
      return canonical(chain[:i+1]) + uppermost(chain[i+1:])
  return chain # NOTE at this point promotion essentially failed, maybe it's better to raise an exception

## TRANSFORM ITEMS

from ._rust import *

stricttransformitem = types.strict[TransformItem]
stricttransform = types.tuple[stricttransformitem]

## EVALUABLE TRANSFORM CHAIN

class EvaluableTransformChain(Evaluable):
  '''The :class:`~nutils.evaluable.Evaluable` equivalent of a transform chain.

  Attributes
  ----------
  todims : :class:`int`
      The to dimension of the transform chain.
  fromdims : :class:`int`
      The from dimension of the transform chain.
  '''

  __slots__ = 'todims', 'fromdims'

  @staticmethod
  def empty(__dim: int) -> 'EvaluableTransformChain':
    '''Return an empty evaluable transform chain with the given dimension.

    Parameters
    ----------
    dim : :class:`int`
        The to and from dimensions of the empty transform chain.

    Returns
    -------
    :class:`EvaluableTransformChain`
        The empty evaluable transform chain.
    '''

    return _EmptyTransformChain(__dim)

  @staticmethod
  def from_argument(name: str, todims: int, fromdims: int) -> 'EvaluableTransformChain':
    '''Return an evaluable transform chain that evaluates to the given argument.

    Parameters
    ----------
    name : :class:`str`
        The name of the argument.
    todims : :class:`int`
        The to dimension of the transform chain.
    fromdims: :class:`int`
        The from dimension of the transform chain.

    Returns
    -------
    :class:`EvaluableTransformChain`
        The transform chain that evaluates to the given argument.
    '''

    return _TransformChainArgument(name, todims, fromdims)

  def __init__(self, args: Tuple[Evaluable, ...], todims: int, fromdims: int) -> None:
    if fromdims > todims:
      raise ValueError('The dimension of the tip cannot be larger than the dimension of the root.')
    self.todims = todims
    self.fromdims = fromdims
    super().__init__(args)

  @property
  def linear(self) -> Array:
    ':class:`nutils.evaluable.Array`: The linear transformation matrix of the entire transform chain. Shape ``(todims,fromdims)``.'

    return _Linear(self)

  @property
  def basis(self) -> Array:
    ':class:`nutils.evaluable.Array`: A basis for the root coordinate system such that the first :attr:`fromdims` vectors span the tangent space. Shape ``(todims,todims)``.'

    if self.fromdims == self.todims:
      return evaluable.diagonalize(evaluable.ones((self.todims,)))
    else:
      return _Basis(self)

  def apply(self, __coords: Array) -> Array:
    '''Apply this transform chain to the last axis given coordinates.

    Parameters
    ----------
    coords : :class:`nutils.evaluable.Array`
        The coordinates to transform with shape ``(...,fromdims)``.

    Returns
    -------
    :class:`nutils.evaluable.Array`
        The transformed coordinates with shape ``(...,todims)``.
    '''

    return _Apply(self, __coords)

  def index_with_tail_in(self, __sequence: 'Transforms') -> Tuple[Array, 'EvaluableTransformChain']:
    '''Return the evaluable index of this transform chain in the given sequence.

    Parameters
    ----------
    sequence : :class:`nutils.transformseq.Transforms`
        The sequence of transform chains.

    Returns
    -------
    :class:`nutils.evaluable.Array`
        The index of this transform chain in the given sequence.
    :class:`EvaluableTransformChain`
        The tail.

    See also
    --------
    :meth:`nutils.transformseq.Transforms.index_with_tail` : the unevaluable version of this method
    '''

    index_tail = _EvaluableIndexWithTail(__sequence, self)
    index = evaluable.ArrayFromTuple(index_tail, 0, (), int, _lower=0, _upper=len(__sequence) - 1)
    tails = _EvaluableTransformChainFromTuple(index_tail, 1, __sequence.fromdims, self.fromdims)
    return index, tails

class _Linear(Array):

  __slots__ = '_fromdims'

  def __init__(self, chain: EvaluableTransformChain) -> None:
    self._fromdims = chain.fromdims
    super().__init__(args=(chain,), shape=(chain.todims, chain.fromdims), dtype=float)

  def evalf(self, chain: TransformChain) -> numpy.ndarray:
    return functools.reduce(lambda r, i: i @ r, (item.linear for item in reversed(chain)), numpy.eye(self._fromdims))

  def _derivative(self, var: evaluable.DerivativeTargetBase, seen: Dict[Evaluable, Evaluable]) -> Array:
    return evaluable.zeros(self.shape + var.shape, dtype=float)

class _Basis(Array):

  __slots__ = '_todims', '_fromdims'

  def __init__(self, chain: EvaluableTransformChain) -> None:
    self._todims = chain.todims
    self._fromdims = chain.fromdims
    super().__init__(args=(chain,), shape=(chain.todims, chain.todims), dtype=float)

  def evalf(self, chain: TransformChain) -> numpy.ndarray:
    linear = numpy.eye(self._fromdims)
    for item in reversed(chain):
      linear = item.linear @ linear
      assert item.fromdims <= item.todims <= item.fromdims + 1
      if item.todims == item.fromdims + 1:
        linear = numpy.concatenate([linear, item.ext[:,_]], axis=1)
    assert linear.shape == (self._todims, self._todims)
    return linear

  def _derivative(self, var: evaluable.DerivativeTargetBase, seen: Dict[Evaluable, Evaluable]) -> Array:
    return evaluable.zeros(self.shape + var.shape, dtype=float)

class _Apply(Array):

  __slots__ = '_chain', '_coords'

  def __init__(self, chain: EvaluableTransformChain, coords: Array) -> None:
    if coords.ndim == 0:
      raise ValueError('expected a coords array with at least one axis but got {}'.format(coords))
    if not evaluable.equalindex(chain.fromdims, coords.shape[-1]):
      raise ValueError('the last axis of coords does not match the from dimension of the transform chain')
    self._chain = chain
    self._coords = coords
    super().__init__(args=(chain, coords), shape=(*coords.shape[:-1], chain.todims), dtype=float)

  def evalf(self, chain: TransformChain, coords: numpy.ndarray) -> numpy.ndarray:
    return apply(chain, coords)

  def _derivative(self, var: evaluable.DerivativeTargetBase, seen: Dict[Evaluable, Evaluable]) -> Array:
    axis = self._coords.ndim - 1
    linear = evaluable.appendaxes(evaluable.prependaxes(self._chain.linear, self._coords.shape[:-1]), var.shape)
    dcoords = evaluable.insertaxis(evaluable.derivative(self._coords, var, seen), axis, linear.shape[axis])
    return evaluable.dot(linear, dcoords, axis+1)

class _EmptyTransformChain(EvaluableTransformChain):

  __slots__ = ()

  def __init__(self, dim: int) -> None:
    super().__init__((), dim, dim)

  def evalf(self) -> TransformChain:
    return ()

  def apply(self, points: Array) -> Array:
    return points

  @property
  def linear(self):
    return evaluable.diagonalize(evaluable.ones((self.todims,)))

class _TransformChainArgument(EvaluableTransformChain):

  __slots__ = '_name'

  def __init__(self, name: str, todims: int, fromdims: int) -> None:
    self._name = name
    super().__init__((evaluable.EVALARGS,), todims, fromdims)

  def evalf(self, evalargs) -> TransformChain:
    chain = evalargs[self._name]
    assert isinstance(chain, tuple) and all(isinstance(item, TransformItem) for item in chain)
    assert not chain or chain[0].todims == self.todims and chain[-1].fromdims == self.fromdims
    return chain

  @property
  def arguments(self):
    return frozenset({self})

class _EvaluableIndexWithTail(evaluable.Evaluable):

  __slots__ = '_sequence'

  def __init__(self, sequence: 'Transforms', chain: EvaluableTransformChain) -> None:
    self._sequence = sequence
    super().__init__((chain,))

  def evalf(self, chain: TransformChain) -> Tuple[numpy.ndarray, TransformChain]:
    index, tails = self._sequence.index_with_tail(chain)
    return numpy.array(index), tails

class _EvaluableTransformChainFromTuple(EvaluableTransformChain):

  __slots__ = '_index'

  def __init__(self, items: evaluable.Evaluable, index: int, todims: int, fromdims: int) -> None:
    self._index = index
    super().__init__((items,), todims, fromdims)

  def evalf(self, items: tuple) -> TransformChain:
    return items[self._index]

# vim:sw=2:sts=2:et
