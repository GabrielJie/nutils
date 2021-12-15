use ndarray::{
    array, concatenate, s, Array1, Array2, ArrayBase, ArrayD, ArrayViewD, ArrayViewMutD, Axis,
    Data, Dimension, Ix2, Slice, Zip,
};
use std::hash::{Hash, Hasher};
use std::{error, fmt};

#[derive(Debug, PartialEq)]
pub enum Error {
    LinearOffsetIncompatibleDimensions,
    NotSquare,
    NotAnUpdim,
    SimplexChildOutOfRange { dim: usize, ichild: usize },
    SimplexEdgeOutOfRange { dim: usize, iedge: usize },
    SimplexChildNotImplemented { dim: usize },
    SimplexEdgeNotImplemented { dim: usize },
    DeterminantNonSquare,
    ExtNotImplemented { trans: TransformItem },
    ExtDoesNotExist,
    TransformPairDimensionMismatch,
    TransformFromDimCoordsMismatch,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LinearOffsetIncompatibleDimensions => {
                write!(f, "Linear and offset have incompatible dimensions.")
            }
            Self::NotSquare => write!(f, "The transform is not an updim."),
            Self::NotAnUpdim => write!(f, "The transform is not an updim."),
            Self::SimplexChildOutOfRange { dim, ichild } => write!(
                f,
                "Child index {} out of range for {}d simplex.",
                ichild, dim
            ),
            Self::SimplexEdgeOutOfRange { dim, iedge } => {
                write!(f, "Edge index {} out of range for {}d simplex.", iedge, dim)
            }
            Self::SimplexChildNotImplemented { dim } => {
                write!(f, "Child transform not implemented for {}d simplex.", dim)
            }
            Self::SimplexEdgeNotImplemented { dim } => {
                write!(f, "Edge transform not implemented for {}d simplex.", dim)
            }
            Self::DeterminantNonSquare => write!(
                f,
                "Cannot compute the determinant for a non-square transform."
            ),
            Self::ExtNotImplemented { trans } => {
                write!(f, "The ext vector is not implement for {:?}.", trans)
            }
            Self::ExtDoesNotExist => {
                f.write_str("This `TransformItem` does not have an ext vector.")
            }
            Self::TransformPairDimensionMismatch => f.write_str("dimension mismatch"),
            Self::TransformFromDimCoordsMismatch => f.write_str("dimension mismatch"),
        }
    }
}

impl error::Error for Error {}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct HashVec<T>(Vec<T>);

impl Hash for HashVec<f64> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for v in self.0.iter() {
            ((*v * 1e6) as u64).hash(state)
        }
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd, Hash)]
pub enum TransformItem {
    Identity(usize),
    Index {
        dim: usize,
        index: isize,
    },
    Point(HashVec<f64>),
    Updim {
        linear: HashVec<f64>,
        offset: HashVec<f64>,
        is_flipped: bool,
    },
    ScaledUpdim(Box<Self>, Box<Self>),
    SimplexEdge10 {
        is_flipped: bool,
    },
    SimplexEdge11 {
        is_flipped: bool,
    },
    SimplexEdge20 {
        is_flipped: bool,
    },
    SimplexEdge21 {
        is_flipped: bool,
    },
    SimplexEdge22 {
        is_flipped: bool,
    },
    SimplexEdge30 {
        is_flipped: bool,
    },
    SimplexEdge31 {
        is_flipped: bool,
    },
    SimplexEdge32 {
        is_flipped: bool,
    },
    SimplexEdge33 {
        is_flipped: bool,
    },
    SimplexChild00,
    SimplexChild10,
    SimplexChild11,
    SimplexChild20,
    SimplexChild21,
    SimplexChild22,
    SimplexChild23,
    SimplexChild30,
    SimplexChild31,
    SimplexChild32,
    SimplexChild33,
    SimplexChild34,
    SimplexChild35,
    SimplexChild36,
    SimplexChild37,
    TensorEdge1(Box<TransformItem>, usize),
    TensorEdge2(usize, Box<TransformItem>),
    TensorChild(Box<TransformItem>, Box<TransformItem>),
}

fn block_diag<S1, S2, T>(block1: ArrayBase<S1, Ix2>, block2: ArrayBase<S2, Ix2>) -> Array2<T>
where
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    T: Clone + num_traits::identities::Zero,
{
    let (m1, n1) = block1.raw_dim().into_pattern();
    let (m2, n2) = block2.raw_dim().into_pattern();
    let mut output = Array2::zeros((m1 + m2, n1 + n2));
    output.slice_mut(s![..m1, ..n1]).assign(&block1);
    output.slice_mut(s![m1.., n1..]).assign(&block2);
    output
}

impl TransformItem {
    pub fn new_identity(dim: usize) -> Self {
        Self::Identity(dim)
    }
    pub fn new_index(dim: usize, index: isize) -> Self {
        Self::Index { dim, index }
    }
    pub fn new_point(offset: Vec<f64>) -> Self {
        Self::Point(HashVec(offset))
    }
    pub fn new_updim(linear: Vec<f64>, offset: Vec<f64>, is_flipped: bool) -> Result<Self, Error> {
        if offset.is_empty() || linear.len() != offset.len() * (offset.len() - 1) {
            Err(Error::NotAnUpdim)
        } else {
            Ok(Self::Updim {
                linear: HashVec(linear),
                offset: HashVec(offset),
                is_flipped,
            })
        }
    }
    pub fn new_scaled_updim(trans1: TransformItem, trans2: TransformItem) -> Result<Self, Error> {
        if trans2.to_dim() != trans1.from_dim() {
            Err(Error::TransformPairDimensionMismatch)
        } else if trans1.to_dim() != trans1.from_dim() {
            Err(Error::NotSquare)
        } else if trans2.to_dim() != trans2.from_dim() + 1 {
            Err(Error::NotAnUpdim)
        } else {
            Ok(Self::ScaledUpdim(Box::new(trans1), Box::new(trans2)))
        }
    }
    pub fn new_simplex_child(dim: usize, ichild: usize) -> Result<Self, Error> {
        match (dim, ichild) {
            (0, 0) => Ok(Self::SimplexChild00),
            (0, _) => Err(Error::SimplexChildOutOfRange { dim, ichild }),
            (1, 0) => Ok(Self::SimplexChild10),
            (1, 1) => Ok(Self::SimplexChild11),
            (1, _) => Err(Error::SimplexChildOutOfRange { dim, ichild }),
            (2, 0) => Ok(Self::SimplexChild20),
            (2, 1) => Ok(Self::SimplexChild21),
            (2, 2) => Ok(Self::SimplexChild22),
            (2, 3) => Ok(Self::SimplexChild23),
            (2, _) => Err(Error::SimplexChildOutOfRange { dim, ichild }),
            (3, 0) => Ok(Self::SimplexChild30),
            (3, 1) => Ok(Self::SimplexChild31),
            (3, 2) => Ok(Self::SimplexChild32),
            (3, 3) => Ok(Self::SimplexChild33),
            (3, 4) => Ok(Self::SimplexChild34),
            (3, 5) => Ok(Self::SimplexChild35),
            (3, 6) => Ok(Self::SimplexChild36),
            (3, 7) => Ok(Self::SimplexChild37),
            (3, _) => Err(Error::SimplexChildOutOfRange { dim, ichild }),
            _ => Err(Error::SimplexChildNotImplemented { dim }),
        }
    }
    pub fn new_simplex_edge(dim: usize, iedge: usize) -> Result<Self, Error> {
        match (dim, iedge) {
            (1, 0) => Ok(Self::SimplexEdge10 { is_flipped: false }),
            (1, 1) => Ok(Self::SimplexEdge11 { is_flipped: false }),
            (1, _) => Err(Error::SimplexEdgeOutOfRange { dim, iedge }),
            (2, 0) => Ok(Self::SimplexEdge20 { is_flipped: false }),
            (2, 1) => Ok(Self::SimplexEdge21 { is_flipped: false }),
            (2, 2) => Ok(Self::SimplexEdge22 { is_flipped: false }),
            (2, _) => Err(Error::SimplexEdgeOutOfRange { dim, iedge }),
            (3, 0) => Ok(Self::SimplexEdge30 { is_flipped: false }),
            (3, 1) => Ok(Self::SimplexEdge31 { is_flipped: false }),
            (3, 2) => Ok(Self::SimplexEdge32 { is_flipped: false }),
            (3, 3) => Ok(Self::SimplexEdge33 { is_flipped: false }),
            (3, _) => Err(Error::SimplexEdgeOutOfRange { dim, iedge }),
            _ => Err(Error::SimplexEdgeNotImplemented { dim }),
        }
    }
    pub fn new_tensor_child(trans1: TransformItem, trans2: TransformItem) -> Result<Self, Error> {
        if trans1.to_dim() != trans1.from_dim() || trans2.to_dim() != trans2.from_dim() {
            Err(Error::NotSquare)
        } else {
            Ok(Self::new_tensor_child_unchecked(trans1, trans2))
        }
    }
    fn new_tensor_child_unchecked(trans1: TransformItem, trans2: TransformItem) -> Self {
        match (trans1.to_dim(), trans2.to_dim()) {
            (0, 0) => Self::SimplexChild00,
            (_, 0) => trans1,
            (0, _) => trans2,
            (_, _) => Self::TensorChild(Box::new(trans1), Box::new(trans2)),
        }
    }
    pub fn new_tensor_edge1(trans1: TransformItem, dim2: usize) -> Result<Self, Error> {
        if trans1.to_dim() != trans1.from_dim() + 1 {
            Err(Error::NotAnUpdim)
        } else {
            Ok(Self::TensorEdge1(Box::new(trans1), dim2))
        }
    }
    pub fn new_tensor_edge2(dim1: usize, trans2: TransformItem) -> Result<Self, Error> {
        if trans2.to_dim() != trans2.from_dim() + 1 {
            Err(Error::NotAnUpdim)
        } else {
            Ok(Self::TensorEdge2(dim1, Box::new(trans2)))
        }
    }
    pub fn to_dim(&self) -> usize {
        match self {
            Self::Identity(dim) | Self::Index { dim, .. } => *dim,
            Self::Point(offset) => offset.0.len(),
            Self::SimplexChild00 => 0,
            Self::SimplexChild10
            | Self::SimplexChild11
            | Self::SimplexEdge10 { .. }
            | Self::SimplexEdge11 { .. } => 1,
            Self::SimplexChild20
            | Self::SimplexChild21
            | Self::SimplexChild22
            | Self::SimplexChild23
            | Self::SimplexEdge20 { .. }
            | Self::SimplexEdge21 { .. }
            | Self::SimplexEdge22 { .. } => 2,
            Self::SimplexChild30
            | Self::SimplexChild31
            | Self::SimplexChild32
            | Self::SimplexChild33
            | Self::SimplexChild34
            | Self::SimplexChild35
            | Self::SimplexChild36
            | Self::SimplexChild37
            | Self::SimplexEdge30 { .. }
            | Self::SimplexEdge31 { .. }
            | Self::SimplexEdge32 { .. }
            | Self::SimplexEdge33 { .. } => 3,
            Self::Updim { offset, .. } => offset.0.len(),
            Self::ScaledUpdim(trans1, _) => trans1.to_dim(),
            Self::TensorChild(trans1, trans2) => trans1.to_dim() + trans2.to_dim(),
            Self::TensorEdge1(trans1, dim2) => trans1.to_dim() + dim2,
            Self::TensorEdge2(dim1, trans2) => dim1 + trans2.to_dim(),
        }
    }
    pub fn from_dim(&self) -> usize {
        match self {
            Self::Identity(dim) | Self::Index { dim, .. } => *dim,
            Self::Point(_) | Self::SimplexChild00 | Self::SimplexEdge10 { .. } | Self::SimplexEdge11 { .. } => 0,
            Self::SimplexChild10
            | Self::SimplexChild11
            | Self::SimplexEdge20 { .. }
            | Self::SimplexEdge21 { .. }
            | Self::SimplexEdge22 { .. } => 1,
            Self::SimplexChild20
            | Self::SimplexChild21
            | Self::SimplexChild22
            | Self::SimplexChild23
            | Self::SimplexEdge30 { .. }
            | Self::SimplexEdge31 { .. }
            | Self::SimplexEdge32 { .. }
            | Self::SimplexEdge33 { .. } => 2,
            Self::SimplexChild30
            | Self::SimplexChild31
            | Self::SimplexChild32
            | Self::SimplexChild33
            | Self::SimplexChild34
            | Self::SimplexChild35
            | Self::SimplexChild36
            | Self::SimplexChild37 => 3,
            Self::Updim { offset, .. } => offset.0.len() - 1,
            Self::ScaledUpdim(_, trans2) => trans2.from_dim(),
            Self::TensorChild(trans1, trans2) => trans1.from_dim() + trans2.from_dim(),
            Self::TensorEdge1(trans1, dim2) => trans1.from_dim() + dim2,
            Self::TensorEdge2(dim1, trans2) => dim1 + trans2.from_dim(),
        }
    }
    pub fn linear(&self) -> Array2<f64> {
        match self {
            Self::Identity(dim) | Self::Index { dim, .. } => Array2::eye(*dim),
            Self::Point(offset) => Array2::zeros((0, offset.0.len())),
            Self::SimplexChild00 => Array2::zeros((0, 0)),
            Self::SimplexChild10 | Self::SimplexChild11 => array![[0.5]],
            Self::SimplexChild20 | Self::SimplexChild21 | Self::SimplexChild22 => {
                array![[0.5, 0.0], [0.0, 0.5]]
            }
            Self::SimplexChild23 => array![[-0.5, 0.0], [0.5, 0.5]],
            Self::SimplexChild30
            | Self::SimplexChild31
            | Self::SimplexChild32
            | Self::SimplexChild33 => array![[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
            Self::SimplexChild34 => array![[-0.5, 0.0, -0.5], [0.5, 0.5, 0.0], [0.0, 0.0, 0.5]],
            Self::SimplexChild35 => array![[0.0, -0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.5]],
            Self::SimplexChild36 => array![[0.5, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.5, 0.5]],
            Self::SimplexChild37 => array![[-0.5, 0.0, -0.5], [-0.5, -0.5, 0.0], [0.5, 0.5, 0.5]],
            Self::SimplexEdge10 { .. } | Self::SimplexEdge11 { .. } => Array2::zeros((1, 0)),
            Self::SimplexEdge20 { .. } => array![[-1.0], [1.0]],
            Self::SimplexEdge21 { .. } => array![[0.0], [1.0]],
            Self::SimplexEdge22 { .. } => array![[1.0], [0.0]],
            Self::SimplexEdge30 { .. } => array![[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]],
            Self::SimplexEdge31 { .. } => array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            Self::SimplexEdge32 { .. } => array![[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
            Self::SimplexEdge33 { .. } => array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            Self::Updim { linear, .. } => Array1::from_vec(linear.0.clone())
                .into_shape((self.to_dim(), self.from_dim()))
                .unwrap(),
            Self::ScaledUpdim(trans1, trans2) => trans1.linear().dot(&trans2.linear()),
            Self::TensorChild(trans1, trans2) => block_diag(trans1.linear(), trans2.linear()),
            Self::TensorEdge1(trans1, dim2) => block_diag(trans1.linear(), Array2::eye(*dim2)),
            Self::TensorEdge2(dim1, trans2) => block_diag(Array2::eye(*dim1), trans2.linear()),
        }
    }
    pub fn inverse_linear(&self) -> Option<Array2<f64>> {
        match self {
            Self::Identity(dim) | Self::Index { dim, .. } => Some(Array2::eye(*dim)),
            Self::SimplexChild00 => Some(Array2::zeros((0, 0))),
            Self::SimplexChild10 | Self::SimplexChild11 => Some(array![[2.]]),
            Self::SimplexChild20 | Self::SimplexChild21 | Self::SimplexChild22 => {
                Some(array![[2., 0.], [0., 2.]])
            }
            Self::SimplexChild23 => Some(array![[-2., -0.], [2., 2.]]),
            Self::SimplexChild30
            | Self::SimplexChild31
            | Self::SimplexChild32
            | Self::SimplexChild33 => Some(array![[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]),
            Self::SimplexChild34 => Some(array![[-2., -0., -2.], [2., 2., 2.], [0., 0., 2.]]),
            Self::SimplexChild35 => Some(array![[0., 2., 0.], [-2., -0., -0.], [2., 0., 2.]]),
            Self::SimplexChild36 => Some(array![[2., 0., 0.], [-0., -2., -0.], [0., 2., 2.]]),
            Self::SimplexChild37 => Some(array![[-2., -2., -2.], [2., -0., 2.], [0., 2., 2.]]),
            Self::TensorChild(trans1, trans2) => {
                if let (Some(linear1), Some(linear2)) =
                    (trans1.inverse_linear(), trans2.inverse_linear())
                {
                    Some(block_diag(linear1, linear2))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
    pub fn offset(&self) -> Array1<f64> {
        match self {
            Self::Identity(dim) | Self::Index { dim, .. } => Array1::zeros((*dim,)),
            Self::Point(offset) => Array1::from_vec(offset.0.clone()),
            Self::SimplexChild00 => Array1::zeros((0,)),
            Self::SimplexChild10 => array![0.0],
            Self::SimplexChild11 => array![0.5],
            Self::SimplexChild20 => array![0.0, 0.0],
            Self::SimplexChild21 | Self::SimplexChild23 => array![0.5, 0.0],
            Self::SimplexChild22 => array![0.0, 0.5],
            Self::SimplexChild30 => array![0.0, 0.0, 0.0],
            Self::SimplexChild31 | Self::SimplexChild34 | Self::SimplexChild35 => {
                array![0.5, 0.0, 0.0]
            }
            Self::SimplexChild32 | Self::SimplexChild36 => array![0.0, 0.5, 0.0],
            Self::SimplexChild33 => array![0.0, 0.0, 0.5],
            Self::SimplexChild37 => array![0.5, 0.5, 0.0],
            Self::SimplexEdge10 { .. } => array![1.0],
            Self::SimplexEdge11 { .. } => array![0.0],
            Self::SimplexEdge20 { .. } => array![1.0, 0.0],
            Self::SimplexEdge21 { .. } | Self::SimplexEdge22 { .. } => array![0.0, 0.0],
            Self::SimplexEdge30 { .. } => array![1.0, 0.0, 0.0],
            Self::SimplexEdge31 { .. }
            | Self::SimplexEdge32 { .. }
            | Self::SimplexEdge33 { .. } => array![0.0, 0.0, 0.0],
            Self::Updim { offset, .. } => Array1::from_vec(offset.0.clone()),
            Self::ScaledUpdim(trans1, trans2) => {
                let mut offset = Array1::zeros(self.to_dim());
                trans1
                    .apply_into(
                        &trans2.offset().view().into_dyn(),
                        &mut offset.view_mut().into_dyn(),
                    )
                    .unwrap();
                offset
            }
            Self::TensorChild(trans1, trans2) => {
                concatenate![Axis(0), trans1.offset(), trans2.offset()]
            }
            Self::TensorEdge1(trans1, dim2) => {
                concatenate![Axis(0), trans1.offset(), Array1::zeros((*dim2,))]
            }
            Self::TensorEdge2(dim1, trans2) => {
                concatenate![Axis(0), Array1::zeros((*dim1,)), trans2.offset()]
            }
        }
    }
    pub fn ext(&self) -> Result<Array1<f64>, Error> {
        match self {
            Self::Identity(_)
            | Self::Index { .. }
            | Self::Point(_)
            | Self::SimplexChild00
            | Self::SimplexChild10
            | Self::SimplexChild11
            | Self::SimplexChild20
            | Self::SimplexChild21
            | Self::SimplexChild22
            | Self::SimplexChild23
            | Self::SimplexChild30
            | Self::SimplexChild31
            | Self::SimplexChild32
            | Self::SimplexChild33
            | Self::SimplexChild34
            | Self::SimplexChild35
            | Self::SimplexChild36
            | Self::SimplexChild37
            | Self::TensorChild(_, _) => Err(Error::ExtDoesNotExist),
            Self::SimplexEdge10 { is_flipped: false }
            | Self::SimplexEdge11 { is_flipped: true } => Ok(array![1.]),
            Self::SimplexEdge10 { is_flipped: true }
            | Self::SimplexEdge11 { is_flipped: false } => Ok(array![-1.]),
            Self::SimplexEdge20 { is_flipped: false } => Ok(array![1., 1.]),
            Self::SimplexEdge20 { is_flipped: true } => Ok(array![-1., -1.]),
            Self::SimplexEdge21 { is_flipped: false } => Ok(array![-1., 0.]),
            Self::SimplexEdge21 { is_flipped: true } => Ok(array![1., 0.]),
            Self::SimplexEdge22 { is_flipped: false } => Ok(array![0., -1.]),
            Self::SimplexEdge22 { is_flipped: true } => Ok(array![0., 1.]),
            Self::SimplexEdge30 { is_flipped: false } => Ok(array![1., 1., 1.]),
            Self::SimplexEdge30 { is_flipped: true } => Ok(array![-1., -1., -1.]),
            Self::SimplexEdge31 { is_flipped: false } => Ok(array![-1., 0., 0.]),
            Self::SimplexEdge31 { is_flipped: true } => Ok(array![1., 0., 0.]),
            Self::SimplexEdge32 { is_flipped: false } => Ok(array![0., -1., 0.]),
            Self::SimplexEdge32 { is_flipped: true } => Ok(array![0., 1., 0.]),
            Self::SimplexEdge33 { is_flipped: false } => Ok(array![0., 0., -1.]),
            Self::SimplexEdge33 { is_flipped: true } => Ok(array![0., 0., 1.]),
            Self::TensorEdge1(trans1, dim2) => {
                Ok(concatenate![Axis(0), trans1.ext()?, Array1::zeros(*dim2)])
            }
            Self::TensorEdge2(dim1, trans2) => {
                Ok(concatenate![Axis(0), Array1::zeros(*dim1), trans2.ext()?])
            }
            _ => (match (self.to_dim(), self.from_dim()) {
                (1, 0) => Ok(array![1.]),
                (2, 1) => {
                    let l = self.linear();
                    Ok(array![l[[1, 0]], -l[[0, 0]]])
                }
                (3, 2) => {
                    let l = self.linear();
                    Ok(array![
                        l[[1, 0]] * l[[2, 1]] - l[[2, 0]] * l[[1, 1]],
                        l[[2, 0]] * l[[0, 1]] - l[[0, 0]] * l[[2, 1]],
                        l[[0, 0]] * l[[1, 1]] - l[[1, 0]] * l[[0, 1]]
                    ])
                }
                (m, n) if m == n + 1 => Err(Error::ExtNotImplemented {
                    trans: self.clone(),
                }),
                _ => Err(Error::ExtDoesNotExist),
            })
            .map(|v| v * (if self.is_flipped() { -1f64 } else { 1f64 })),
        }
    }
    pub fn apply(&self, from_coords: &ArrayViewD<f64>) -> Result<ArrayD<f64>, Error> {
        let mut to_shape = from_coords.shape().to_vec();
        to_shape.pop();
        to_shape.push(self.to_dim());
        let mut to_coords: ArrayD<f64> = ArrayD::zeros(to_shape);
        self.apply_into(from_coords, &mut to_coords.view_mut())?;
        Ok(to_coords)
    }
    pub fn apply_into(
        &self,
        from_coords: &ArrayViewD<f64>,
        to_coords: &mut ArrayViewMutD<f64>,
    ) -> Result<(), Error> {
        if from_coords.shape().last() != Some(&self.from_dim())
            || to_coords.shape().last() != Some(&self.to_dim())
        {
            return Err(Error::TransformFromDimCoordsMismatch);
        }
        match self {
            Self::Identity(_) | Self::Index { .. } => to_coords.assign(from_coords),
            Self::Point(_) => to_coords.assign(&self.offset()),
            Self::SimplexChild00 => {}
            Self::SimplexChild10 => to_coords.zip_mut_with(from_coords, |t, f| *t = 0.5 * f),
            Self::SimplexChild11 => to_coords.zip_mut_with(from_coords, |t, f| *t = 0.5 * f + 0.5),
            Self::SimplexEdge10 { .. } => to_coords.fill(1.),
            Self::SimplexEdge11 { .. } => to_coords.fill(0.),
            Self::TensorChild(trans1, trans2) => {
                let axis = Axis(from_coords.ndim() - 1);
                trans1.apply_into(
                    &from_coords.slice_axis(axis, Slice::from(..trans1.from_dim())),
                    &mut to_coords.slice_axis_mut(axis, Slice::from(..trans1.to_dim())),
                )?;
                trans2.apply_into(
                    &from_coords.slice_axis(axis, Slice::from(trans1.from_dim()..)),
                    &mut to_coords.slice_axis_mut(axis, Slice::from(trans1.to_dim()..)),
                )?;
            }
            Self::TensorEdge1(trans1, _) => {
                let axis = Axis(from_coords.ndim() - 1);
                trans1.apply_into(
                    &from_coords.slice_axis(axis, Slice::from(..trans1.from_dim())),
                    &mut to_coords.slice_axis_mut(axis, Slice::from(..trans1.to_dim())),
                )?;
                to_coords
                    .slice_axis_mut(axis, Slice::from(trans1.to_dim()..))
                    .assign(&from_coords.slice_axis(axis, Slice::from(trans1.from_dim()..)));
            }
            Self::TensorEdge2(dim1, trans2) => {
                let axis = Axis(from_coords.ndim() - 1);
                to_coords
                    .slice_axis_mut(axis, Slice::from(..*dim1))
                    .assign(&from_coords.slice_axis(axis, Slice::from(..*dim1)));
                trans2.apply_into(
                    &from_coords.slice_axis(axis, Slice::from(*dim1..)),
                    &mut to_coords.slice_axis_mut(axis, Slice::from(*dim1..)),
                )?;
            }
            _ => {
                let linear = self.linear();
                let offset = self.offset();
                Zip::from(to_coords.rows_mut())
                    .and(from_coords.rows())
                    .for_each(|to_i, from_i| {
                        Zip::from(to_i).and(&offset).and(linear.rows()).for_each(
                            |to_ij, &offset_j, linear_j| *to_ij = offset_j + linear_j.dot(&from_i),
                        )
                    });
            }
        };
        Ok(())
    }
    pub fn unapply(&self, to_coords: &ArrayViewD<f64>) -> Result<ArrayD<f64>, Error> {
        let mut from_shape = to_coords.shape().to_vec();
        from_shape.pop();
        from_shape.push(self.from_dim());
        let mut from_coords: ArrayD<f64> = ArrayD::zeros(from_shape);
        self.unapply_into(to_coords, &mut from_coords.view_mut())?;
        Ok(from_coords)
    }
    pub fn unapply_into(
        &self,
        to_coords: &ArrayViewD<f64>,
        from_coords: &mut ArrayViewMutD<f64>,
    ) -> Result<(), Error> {
        if from_coords.shape().last() != Some(&self.from_dim())
            || to_coords.shape().last() != Some(&self.to_dim())
        {
            Err(Error::TransformFromDimCoordsMismatch)
        } else if let Some(linear) = self.inverse_linear() {
            let offset = self.offset();
            Zip::from(from_coords.rows_mut())
                .and(to_coords.rows())
                .for_each(|mut from_i, to_i| from_i.assign(&linear.dot(&(&to_i - &offset))));
            Ok(())
        } else {
            Err(Error::NotSquare)
        }
    }
    pub fn transform_polynomial(&self, from_poly: &ArrayViewD<f64>) -> ArrayD<f64> {
        let mut to_poly = from_poly.to_owned();
        let axis = from_poly.ndim() - self.from_dim();
        self.transform_polynomial_inplace(&mut to_poly.view_mut(), axis);
        to_poly
    }
    pub fn transform_polynomial_inplace(&self, poly: &mut ArrayViewMutD<f64>, axis: usize) {
        match self {
            Self::Identity(_) | Self::Index { .. } | Self::SimplexChild00 => {}
            Self::SimplexChild10 => {
                for mut poly_i in poly.lanes_mut(Axis(axis)) {
                    for j in 0..poly_i.len() {
                        poly_i[j] *= 0.5f64.powi(j as i32);
                    }
                }
            }
            Self::SimplexChild11 => {
                let m = poly.shape()[axis];
                let mut work: Array1<f64> = Array1::zeros(m);
                for mut poly_i in poly.lanes_mut(Axis(axis)) {
                    for j in 0..m {
                        poly_i[j] *= 0.5f64.powi(j as i32);
                        work[j] = poly_i[j];
                    }
                    for n in 1..m {
                        let mut binom = 1;
                        poly_i[0] += work[n];
                        for k in 1..n {
                            binom *= (n + 1 - k) / k;
                            poly_i[k] += work[n] * (binom as f64);
                        }
                    }
                }
            }
            Self::TensorChild(trans1, trans2) => {
                trans1.transform_polynomial_inplace(poly, axis);
                trans2.transform_polynomial_inplace(poly, axis + trans1.from_dim());
            }
            _ => panic!("not implemented"),
        };
    }
    pub fn flipped(&self) -> Self {
        match self {
            Self::Identity(_)
            | Self::Index { .. }
            | Self::Point(_)
            | Self::SimplexChild00
            | Self::SimplexChild10
            | Self::SimplexChild11
            | Self::SimplexChild20
            | Self::SimplexChild21
            | Self::SimplexChild22
            | Self::SimplexChild23
            | Self::SimplexChild30
            | Self::SimplexChild31
            | Self::SimplexChild32
            | Self::SimplexChild33
            | Self::SimplexChild34
            | Self::SimplexChild35
            | Self::SimplexChild36
            | Self::SimplexChild37
            | Self::TensorChild(_, _) => self.clone(),
            Self::SimplexEdge10 { is_flipped } => Self::SimplexEdge10 {
                is_flipped: !is_flipped,
            },
            Self::SimplexEdge11 { is_flipped } => Self::SimplexEdge11 {
                is_flipped: !is_flipped,
            },
            Self::SimplexEdge20 { is_flipped } => Self::SimplexEdge20 {
                is_flipped: !is_flipped,
            },
            Self::SimplexEdge21 { is_flipped } => Self::SimplexEdge21 {
                is_flipped: !is_flipped,
            },
            Self::SimplexEdge22 { is_flipped } => Self::SimplexEdge22 {
                is_flipped: !is_flipped,
            },
            Self::SimplexEdge30 { is_flipped } => Self::SimplexEdge30 {
                is_flipped: !is_flipped,
            },
            Self::SimplexEdge31 { is_flipped } => Self::SimplexEdge31 {
                is_flipped: !is_flipped,
            },
            Self::SimplexEdge32 { is_flipped } => Self::SimplexEdge32 {
                is_flipped: !is_flipped,
            },
            Self::SimplexEdge33 { is_flipped } => Self::SimplexEdge33 {
                is_flipped: !is_flipped,
            },
            Self::Updim {
                linear,
                offset,
                is_flipped,
            } => Self::Updim {
                linear: linear.clone(),
                offset: offset.clone(),
                is_flipped: !is_flipped,
            },
            Self::ScaledUpdim(trans1, trans2) => {
                Self::ScaledUpdim(trans1.clone(), Box::new(trans2.flipped()))
            }
            Self::TensorEdge1(trans1, dim2) => Self::TensorEdge1(Box::new(trans1.flipped()), *dim2),
            Self::TensorEdge2(dim1, trans2) => Self::TensorEdge2(*dim1, Box::new(trans2.flipped())),
        }
    }
    pub fn is_flipped(&self) -> bool {
        match self {
            Self::Identity(_)
            | Self::Index { .. }
            | Self::Point(_)
            | Self::SimplexChild00
            | Self::SimplexChild10
            | Self::SimplexChild11
            | Self::SimplexChild20
            | Self::SimplexChild21
            | Self::SimplexChild22
            | Self::SimplexChild30
            | Self::SimplexChild31
            | Self::SimplexChild32
            | Self::SimplexChild33
            | Self::SimplexChild35
            | Self::SimplexChild37
            | Self::TensorChild(_, _) => false,
            Self::SimplexChild23 | Self::SimplexChild34 | Self::SimplexChild36 => true,
            Self::SimplexEdge10 { is_flipped }
            | Self::SimplexEdge11 { is_flipped }
            | Self::SimplexEdge20 { is_flipped }
            | Self::SimplexEdge21 { is_flipped }
            | Self::SimplexEdge22 { is_flipped }
            | Self::SimplexEdge30 { is_flipped }
            | Self::SimplexEdge31 { is_flipped }
            | Self::SimplexEdge32 { is_flipped }
            | Self::SimplexEdge33 { is_flipped }
            | Self::Updim { is_flipped, .. } => *is_flipped,
            Self::ScaledUpdim(trans1, trans2) => trans1.is_flipped() ^ trans2.is_flipped(),
            _ => {
                if let Some(det) = self.det() {
                    det < 0.
                } else {
                    false
                }
            }
        }
    }
    pub fn det(&self) -> Option<f64> {
        match self {
            Self::Identity(_) | Self::Index { .. } | Self::SimplexChild00 => Some(1.),
            Self::SimplexChild10 | Self::SimplexChild11 => Some(0.5),
            Self::SimplexChild20 | Self::SimplexChild21 | Self::SimplexChild22 => Some(0.25),
            Self::SimplexChild23 => Some(-0.25),
            Self::SimplexChild30
            | Self::SimplexChild31
            | Self::SimplexChild32
            | Self::SimplexChild33
            | Self::SimplexChild35
            | Self::SimplexChild37 => Some(0.125),
            Self::SimplexChild34 | Self::SimplexChild36 => Some(-0.125),
            Self::TensorChild(trans1, trans2) => {
                // Note: unwrap is safe because both `trans1` and `trans2` are square.
                Some(trans1.det().unwrap() * trans2.det().unwrap())
            }
            Self::Point(_)
            | Self::SimplexEdge10 { .. }
            | Self::SimplexEdge11 { .. }
            | Self::SimplexEdge20 { .. }
            | Self::SimplexEdge21 { .. }
            | Self::SimplexEdge22 { .. }
            | Self::SimplexEdge30 { .. }
            | Self::SimplexEdge31 { .. }
            | Self::SimplexEdge32 { .. }
            | Self::SimplexEdge33 { .. }
            | Self::TensorEdge1(_, _)
            | Self::TensorEdge2(_, _)
            | Self::Updim { .. }
            | Self::ScaledUpdim(_, _) => None,
            // _ => {
            //     if self.to_dim() != self.from_dim() {
            //         None
            //     } else if self.to_dim() == 0 {
            //         Some(1.)
            //     } else {
            //         Some(self.linear().det().unwrap())
            //     }
            // }
        }
    }
    fn is_child(&self) -> bool {
        match self {
            Self::Identity(_)
            | Self::Index { .. }
            | Self::Point(_)
            | Self::SimplexEdge10 { .. }
            | Self::SimplexEdge11 { .. }
            | Self::SimplexEdge20 { .. }
            | Self::SimplexEdge21 { .. }
            | Self::SimplexEdge22 { .. }
            | Self::SimplexEdge30 { .. }
            | Self::SimplexEdge31 { .. }
            | Self::SimplexEdge32 { .. }
            | Self::SimplexEdge33 { .. }
            | Self::TensorEdge1(_, _)
            | Self::TensorEdge2(_, _)
            | Self::Updim { .. }
            | Self::ScaledUpdim(_, _) => false,
            Self::SimplexChild00
            | Self::SimplexChild10
            | Self::SimplexChild11
            | Self::SimplexChild20
            | Self::SimplexChild21
            | Self::SimplexChild22
            | Self::SimplexChild23
            | Self::SimplexChild30
            | Self::SimplexChild31
            | Self::SimplexChild32
            | Self::SimplexChild33
            | Self::SimplexChild34
            | Self::SimplexChild35
            | Self::SimplexChild36
            | Self::SimplexChild37
            | Self::TensorChild(_, _) => true,
        }
    }
    fn is_edge(&self) -> bool {
        match self {
            Self::Identity(_)
            | Self::Index { .. }
            | Self::Point(_)
            | Self::SimplexChild00
            | Self::SimplexChild10
            | Self::SimplexChild11
            | Self::SimplexChild20
            | Self::SimplexChild21
            | Self::SimplexChild22
            | Self::SimplexChild23
            | Self::SimplexChild30
            | Self::SimplexChild31
            | Self::SimplexChild32
            | Self::SimplexChild33
            | Self::SimplexChild34
            | Self::SimplexChild35
            | Self::SimplexChild36
            | Self::SimplexChild37
            | Self::TensorChild(_, _) => false,
            Self::SimplexEdge10 { .. }
            | Self::SimplexEdge11 { .. }
            | Self::SimplexEdge20 { .. }
            | Self::SimplexEdge21 { .. }
            | Self::SimplexEdge22 { .. }
            | Self::SimplexEdge30 { .. }
            | Self::SimplexEdge31 { .. }
            | Self::SimplexEdge32 { .. }
            | Self::SimplexEdge33 { .. }
            | Self::TensorEdge1(_, _)
            | Self::TensorEdge2(_, _)
            | Self::Updim { .. }
            | Self::ScaledUpdim(_, _) => true,
        }
    }
}

macro_rules! impl_swaps {
    (
        simplex { $($e1:ident + $c1:ident <=> $c2:ident + $e2:ident,)* }
        up { $($up:tt)* }
        down { $($down:tt)* }
    ) => {
        impl TransformItem {
            pub fn swap_up(&self, other: &Self) -> Result<Option<(Self, Self)>, Error> {
                if self.from_dim() != other.to_dim() {
                    return Err(Error::TransformPairDimensionMismatch);
                }
                let result = Ok(match (self, other) {
                    $(
                        (Self::$e1 { is_flipped }, Self::$c1) => {
                            Some((Self::$c2, Self::$e2 { is_flipped: *is_flipped }))
                        }
                    )*
                    $($up)*
                    _ => None,
                });
                // println!("swap_up {:?} -> {:?}", (self, other), result);
                result
            }
            pub fn swap_down(&self, other: &Self) -> Result<Option<(Self, Self)>, Error> {
                if other.from_dim() != self.to_dim() {
                    return Err(Error::TransformPairDimensionMismatch);
                }
                let result = Ok(match (self, other) {
                    $(
                        (Self::$e2 { is_flipped }, Self::$c2) => {
                            Some((Self::$e1 { is_flipped: *is_flipped }, Self::$c1))
                        }
                    )*
                    $($down)*
                    _ => None,
                });
                // println!("swap_down {:?} -> {:?}", (self, other), result);
                result
            }
        }
    };
}

impl_swaps! {
    simplex {
        SimplexEdge10 + SimplexChild00 <=> SimplexChild11 + SimplexEdge10,
        SimplexEdge11 + SimplexChild00 <=> SimplexChild10 + SimplexEdge11,
        SimplexEdge20 + SimplexChild10 <=> SimplexChild21 + SimplexEdge20,
        SimplexEdge20 + SimplexChild11 <=> SimplexChild22 + SimplexEdge20,
        SimplexEdge21 + SimplexChild10 <=> SimplexChild20 + SimplexEdge21,
        SimplexEdge21 + SimplexChild11 <=> SimplexChild22 + SimplexEdge21,
        SimplexEdge22 + SimplexChild10 <=> SimplexChild20 + SimplexEdge22,
        SimplexEdge22 + SimplexChild11 <=> SimplexChild21 + SimplexEdge22,
        SimplexEdge30 + SimplexChild20 <=> SimplexChild31 + SimplexEdge30,
        SimplexEdge30 + SimplexChild21 <=> SimplexChild32 + SimplexEdge30,
        SimplexEdge30 + SimplexChild22 <=> SimplexChild33 + SimplexEdge30,
        SimplexEdge30 + SimplexChild23 <=> SimplexChild37 + SimplexEdge31,
        SimplexEdge31 + SimplexChild20 <=> SimplexChild30 + SimplexEdge31,
        SimplexEdge31 + SimplexChild21 <=> SimplexChild32 + SimplexEdge31,
        SimplexEdge31 + SimplexChild22 <=> SimplexChild33 + SimplexEdge31,
        SimplexEdge31 + SimplexChild23 <=> SimplexChild36 + SimplexEdge31,
        SimplexEdge32 + SimplexChild20 <=> SimplexChild30 + SimplexEdge32,
        SimplexEdge32 + SimplexChild21 <=> SimplexChild31 + SimplexEdge32,
        SimplexEdge32 + SimplexChild22 <=> SimplexChild33 + SimplexEdge32,
        SimplexEdge32 + SimplexChild23 <=> SimplexChild35 + SimplexEdge31,
        SimplexEdge33 + SimplexChild20 <=> SimplexChild30 + SimplexEdge33,
        SimplexEdge33 + SimplexChild21 <=> SimplexChild31 + SimplexEdge33,
        SimplexEdge33 + SimplexChild22 <=> SimplexChild32 + SimplexEdge33,
        SimplexEdge33 + SimplexChild23 <=> SimplexChild34 + SimplexEdge33,
    }
    up {
        (Self::ScaledUpdim(trans1, trans2), Self::Identity(_)) => {
            Some((*trans1.clone(), *trans2.clone()))
        }
        (Self::TensorEdge1(edge1, dim), other) if edge1.from_dim() == 0 && other.is_child() => {
            if let Ok(Some((child2, edge2))) = edge1.swap_up(&Self::SimplexChild00) {
                Some((
                    Self::new_tensor_child_unchecked(child2, other.clone()),
                    Self::TensorEdge1(Box::new(edge2), *dim),
                ))
            } else {
                None
            }
        }
        (Self::TensorEdge1(edge1, dim), Self::TensorChild(child1, other)) => {
            if let Ok(Some((child2, edge2))) = edge1.swap_up(child1) {
                Some((
                    Self::new_tensor_child_unchecked(child2, other.as_ref().clone()),
                    Self::TensorEdge1(Box::new(edge2), *dim),
                ))
            } else {
                None
            }
        }
        (Self::TensorEdge2(dim, edge1), other) if edge1.from_dim() == 0 && other.is_child() => {
            if let Ok(Some((child2, edge2))) = edge1.swap_up(&Self::SimplexChild00) {
                Some((
                    Self::new_tensor_child_unchecked(other.clone(), child2),
                    Self::TensorEdge2(*dim, Box::new(edge2)),
                ))
            } else {
                None
            }
        }
        (Self::TensorEdge2(dim, edge1), Self::TensorChild(other, child1)) => {
            if let Ok(Some((child2, edge2))) = edge1.swap_up(child1) {
                Some((
                    Self::new_tensor_child_unchecked(other.as_ref().clone(), child2),
                    Self::TensorEdge2(*dim, Box::new(edge2)),
                ))
            } else {
                None
            }
        }
    }
    down {
        (t1 @ Self::TensorEdge1(edge1, _), t2 @ Self::TensorChild(child1, other)) => {
            if let Ok(Some((edge2, child2))) = edge1.swap_down(child1) {
                Some((
                    Self::TensorEdge1(Box::new(edge2), other.from_dim()),
                    Self::new_tensor_child_unchecked(child2, other.as_ref().clone()),
                ))
            } else {
                Some((
                    Self::ScaledUpdim(Box::new(t2.clone()), Box::new(t1.clone())),
                    Self::Identity(t1.from_dim()),
                ))
            }
        }
        (t1 @ Self::TensorEdge2(_, edge1), t2 @ Self::TensorChild(other, child1)) => {
            if let Ok(Some((edge2, child2))) = edge1.swap_down(child1) {
                Some((
                    Self::TensorEdge2(other.from_dim(), Box::new(edge2)),
                    Self::new_tensor_child_unchecked(other.as_ref().clone(), child2),
                ))
            } else {
                Some((
                    Self::ScaledUpdim(Box::new(t2.clone()), Box::new(t1.clone())),
                    Self::Identity(t1.from_dim()),
                ))
            }
        }
        (trans1, trans2 @ Self::TensorChild(_, _)) if trans1.is_edge() => {
            Some((
                Self::ScaledUpdim(Box::new(trans2.clone()), Box::new(trans1.clone())),
                Self::Identity(trans1.from_dim()),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_identity() {
        let trans = TransformItem::new_identity(2);
        assert_eq!(trans.to_dim(), 2);
        assert_eq!(trans.from_dim(), 2);
        assert_abs_diff_eq!(trans.apply(&array![1., 2.]).unwrap(), array![1., 2.]);
        assert_abs_diff_eq!(trans.det().unwrap(), 1.);
        assert_eq!(trans.ext(), Err(Error::ExtDoesNotExist));
    }

    #[test]
    fn test_index() {
        let trans = TransformItem::new_index(2, 0);
        assert_eq!(trans.to_dim(), 2);
        assert_eq!(trans.from_dim(), 2);
        assert_abs_diff_eq!(trans.apply(&array![1., 2.]).unwrap(), array![1., 2.]);
        assert_abs_diff_eq!(trans.det().unwrap(), 1.);
        assert_eq!(trans.ext(), Err(Error::ExtDoesNotExist));
    }

    #[test]
    fn test_simplex_child_1_0() {
        let trans = TransformItem::new_simplex_child(1, 0).unwrap();
        assert_eq!(trans.to_dim(), 1);
        assert_eq!(trans.from_dim(), 1);
        assert_abs_diff_eq!(trans.linear(), array![[0.5]]);
        assert_abs_diff_eq!(trans.offset(), array![0.0]);
        assert_abs_diff_eq!(trans.apply(&array![2.]).unwrap(), array![1.]);
        assert_abs_diff_eq!(trans.det().unwrap(), 0.5);
        assert_eq!(trans.ext(), Err(Error::ExtDoesNotExist));
    }

    #[test]
    fn test_simplex_child_1_1() {
        let trans = TransformItem::new_simplex_child(1, 1).unwrap();
        assert_eq!(trans.to_dim(), 1);
        assert_eq!(trans.from_dim(), 1);
        assert_abs_diff_eq!(trans.linear(), array![[0.5]]);
        assert_abs_diff_eq!(trans.offset(), array![0.5]);
        assert_abs_diff_eq!(trans.apply(&array![2.]).unwrap(), array![1.5]);
        assert_abs_diff_eq!(trans.det().unwrap(), 0.5);
        assert_eq!(trans.ext(), Err(Error::ExtDoesNotExist));
    }

    #[test]
    fn test_tensor_child() {
        let trans1 = TransformItem::new_index(2, 0);
        let trans2 = TransformItem::new_index(1, 1);
        let trans = TransformItem::new_tensor_child(trans1, trans2).unwrap();
        assert_eq!(trans.to_dim(), 3);
        assert_eq!(trans.from_dim(), 3);
        assert_abs_diff_eq!(
            trans.apply(&array![1., 2., 3.]).unwrap(),
            array![1., 2., 3.]
        );
        assert_abs_diff_eq!(trans.det().unwrap(), 1.);
        assert_eq!(trans.ext(), Err(Error::ExtDoesNotExist));
    }

    #[test]
    fn test_updim() {
        let trans = TransformItem::new_updim(vec![1., 0.], vec![0., 1.], false).unwrap();
        assert_eq!(trans.to_dim(), 2);
        assert_eq!(trans.from_dim(), 1);
        assert_abs_diff_eq!(trans.apply(&array![2.]).unwrap(), array![2., 1.]);
        assert_eq!(trans.det(), None);
        assert_abs_diff_eq!(trans.ext().unwrap(), array![0., -1.]);
    }

    #[test]
    fn test_scaled_updim() {
        let trans1 = TransformItem::new_identity(2);
        let trans2 = TransformItem::new_updim(vec![1., 0.], vec![0., 1.], false).unwrap();
        let trans = TransformItem::new_scaled_updim(trans1, trans2).unwrap();
        assert_eq!(trans.to_dim(), 2);
        assert_eq!(trans.from_dim(), 1);
        assert_abs_diff_eq!(trans.apply(&array![2.]).unwrap(), array![2., 1.]);
        assert_eq!(trans.det(), None);
        assert_abs_diff_eq!(trans.ext().unwrap(), array![0., -1.]);
    }
}
