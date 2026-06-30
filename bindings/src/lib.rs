// PyO3 Python bindings exposing options-core for prototyping and validation.
use pyo3::prelude::*;

#[pymodule]
fn options_bindings(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
