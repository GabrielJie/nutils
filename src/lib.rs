use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn,
};
use pyo3::class::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

mod transform;

use transform::TransformItem;

impl From<transform::Error> for PyErr {
    fn from(err: transform::Error) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

#[pymodule]
#[allow(non_snake_case)]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyclass(name = "TransformItem", module = "nutils._rust")]
    #[derive(Debug, Clone)]
    struct PyTransformItem(TransformItem);

    #[pymethods]
    impl PyTransformItem {
        pub fn __repr__(&self) -> String {
            format!("{:?}", self.0)
        }
        pub fn __hash__(&self) -> u64 {
            let mut hasher = DefaultHasher::new();
            self.0.hash(&mut hasher);
            hasher.finish()
        }
        pub fn __richcmp__<'py>(
            &self,
            py: Python<'py>,
            other: &'py PyAny,
            op: CompareOp,
        ) -> PyObject {
            if let Ok(PyTransformItem(other)) = other.extract() {
                match op {
                    CompareOp::Lt => self.0 < other,
                    CompareOp::Le => self.0 <= other,
                    CompareOp::Eq => self.0 == other,
                    CompareOp::Ne => self.0 != other,
                    CompareOp::Gt => self.0 > other,
                    CompareOp::Ge => self.0 >= other,
                }
                .into_py(py)
            } else {
                match op {
                    CompareOp::Eq | CompareOp::Ne => false.into_py(py),
                    CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge => {
                        py.NotImplemented()
                    }
                }
            }
        }
        #[getter]
        pub fn todims(&self) -> usize {
            self.0.to_dim()
        }
        #[getter]
        pub fn fromdims(&self) -> usize {
            self.0.from_dim()
        }
        #[getter]
        pub fn linear<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self.0.linear().into_pyarray(py)
        }
        #[getter]
        pub fn offset<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
            self.0.offset().into_pyarray(py)
        }
        #[getter]
        pub fn ext<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
            Ok(self.0.ext()?.into_pyarray(py))
        }
        pub fn apply<'py>(
            &self,
            py: Python<'py>,
            from_coords: PyReadonlyArrayDyn<f64>,
        ) -> PyResult<&'py PyArrayDyn<f64>> {
            Ok(self
                .0
                .apply(&from_coords.as_array().view())?
                .into_pyarray(py))
        }
        pub fn invapply<'py>(
            &self,
            py: Python<'py>,
            to_coords: PyReadonlyArrayDyn<f64>,
        ) -> PyResult<&'py PyArrayDyn<f64>> {
            Ok(self
                .0
                .unapply(&to_coords.as_array().view())?
                .into_pyarray(py))
        }
        #[getter]
        pub fn flipped(&self) -> Self {
            Self(self.0.flipped())
        }
        #[getter]
        pub fn isflipped(&self) -> bool {
            self.0.is_flipped()
        }
        #[getter]
        pub fn det(&self) -> PyResult<f64> {
            if let Some(det) = self.0.det() {
                Ok(det)
            } else {
                Err(PyValueError::new_err("determinant is not defined"))
            }
        }
        pub fn swapup(
            &self,
            other: &PyTransformItem,
        ) -> PyResult<Option<(PyTransformItem, PyTransformItem)>> {
            Ok(self.0.swap_up(&other.0)?.map(|(l, r)| (l.into(), r.into())))
        }
        pub fn swapdown(
            &self,
            other: &PyTransformItem,
        ) -> PyResult<Option<(PyTransformItem, PyTransformItem)>> {
            Ok(self
                .0
                .swap_down(&other.0)?
                .map(|(l, r)| (l.into(), r.into())))
        }
        pub fn transform_poly<'py>(
            &self,
            py: Python<'py>,
            from_poly: PyReadonlyArrayDyn<f64>,
        ) -> &'py PyArrayDyn<f64> {
            self.0.transform_polynomial(&from_poly.as_array().view()).into_pyarray(py)
        }
        pub fn as_index(&self) -> Option<isize> {
            if let TransformItem::Index { index, .. } = self.0 {
                Some(index)
            } else {
                None
            }
        }
    }

    impl From<TransformItem> for PyTransformItem {
        fn from(trans: TransformItem) -> Self {
            Self(trans)
        }
    }

    #[pyfunction]
    fn Identity(dim: usize) -> PyTransformItem {
        TransformItem::new_identity(dim).into()
    }
    #[pyfunction]
    fn Index(dim: usize, index: isize) -> PyTransformItem {
        TransformItem::new_index(dim, index).into()
    }
    #[pyfunction]
    fn Point(offset: PyReadonlyArray1<f64>) -> PyTransformItem {
        TransformItem::new_point(offset.as_array().iter().copied().collect()).into()
    }
    #[pyfunction]
    fn Updim(
        linear: PyReadonlyArray2<f64>,
        offset: PyReadonlyArray1<f64>,
        isflipped: bool,
    ) -> PyResult<PyTransformItem> {
        Ok(TransformItem::new_updim(
            linear.as_array().iter().copied().collect(),
            offset.as_array().iter().copied().collect(),
            isflipped,
        )?
        .into())
    }
    #[pyfunction]
    fn ScaledUpdim(trans1: PyTransformItem, trans2: PyTransformItem) -> PyResult<PyTransformItem> {
        Ok(TransformItem::new_scaled_updim(trans1.0, trans2.0)?.into())
    }
    #[pyfunction]
    fn SimplexChild(dim: usize, ichild: usize) -> PyResult<PyTransformItem> {
        Ok(TransformItem::new_simplex_child(dim, ichild)?.into())
    }
    #[pyfunction]
    fn SimplexEdge(dim: usize, iedge: usize) -> PyResult<PyTransformItem> {
        Ok(TransformItem::new_simplex_edge(dim, iedge)?.into())
    }
    #[pyfunction]
    fn TensorChild(trans1: PyTransformItem, trans2: PyTransformItem) -> PyResult<PyTransformItem> {
        Ok(TransformItem::new_tensor_child(trans1.0, trans2.0)?.into())
    }
    #[pyfunction]
    fn TensorEdge1(trans1: PyTransformItem, dim2: usize) -> PyResult<PyTransformItem> {
        Ok(TransformItem::new_tensor_edge1(trans1.0, dim2)?.into())
    }
    #[pyfunction]
    fn TensorEdge2(dim1: usize, trans2: PyTransformItem) -> PyResult<PyTransformItem> {
        Ok(TransformItem::new_tensor_edge2(dim1, trans2.0)?.into())
    }

    m.add_class::<PyTransformItem>()?;
    m.add_function(wrap_pyfunction!(Identity, m)?)?;
    m.add_function(wrap_pyfunction!(Index, m)?)?;
    m.add_function(wrap_pyfunction!(Point, m)?)?;
    m.add_function(wrap_pyfunction!(Updim, m)?)?;
    m.add_function(wrap_pyfunction!(ScaledUpdim, m)?)?;
    m.add_function(wrap_pyfunction!(SimplexChild, m)?)?;
    m.add_function(wrap_pyfunction!(SimplexEdge, m)?)?;
    m.add_function(wrap_pyfunction!(TensorChild, m)?)?;
    m.add_function(wrap_pyfunction!(TensorEdge1, m)?)?;
    m.add_function(wrap_pyfunction!(TensorEdge2, m)?)?;
    Ok(())
}
