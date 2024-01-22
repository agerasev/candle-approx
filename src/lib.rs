pub use approx;
use approx::{AbsDiffEq, RelativeEq};
use candle_core::{Result, Tensor};
use std::fmt::{self, Debug, Display};

const DEFAULT_EPSILON: f64 = 1e-4;
const DEFAULT_MAX_RELATIVE: f64 = 1e-4;

fn abs_diff_eq(a: &Tensor, b: &Tensor, epsilon: f64) -> Result<Tensor> {
    a.sub(b)?.abs()?.le(epsilon)
}

fn relative_eq(a: &Tensor, b: &Tensor, epsilon: f64, max_relative: f64) -> Result<Tensor> {
    let norm2 = a.abs()?.add(&b.abs()?)?;
    let diff = a.sub(b)?.abs()?;
    diff.le(&(norm2 * (0.5 * max_relative))?.maximum(epsilon)?)
}

fn all(a: &Tensor) -> Result<bool> {
    Ok(a.flatten_all()?.min(0)?.to_scalar::<u8>()? != 0)
}

#[repr(transparent)]
pub struct PanickingTensor(pub Tensor);

impl PartialEq for PanickingTensor {
    fn eq(&self, other: &Self) -> bool {
        all(&self.0.eq(&other.0).unwrap()).unwrap()
    }
}

impl AbsDiffEq for PanickingTensor {
    type Epsilon = f64;
    fn default_epsilon() -> Self::Epsilon {
        DEFAULT_EPSILON
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        all(&abs_diff_eq(&self.0, &other.0, epsilon).unwrap()).unwrap()
    }
}

impl RelativeEq for PanickingTensor {
    fn default_max_relative() -> Self::Epsilon {
        DEFAULT_MAX_RELATIVE
    }
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        all(&relative_eq(&self.0, &other.0, epsilon, max_relative).unwrap()).unwrap()
    }
}

impl Debug for PanickingTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

#[macro_export]
macro_rules! assert_abs_diff_eq {
    ($a:expr, $b:expr) => {
        $crate::approx::assert_abs_diff_eq!(
            $crate::PanickingTensor($a),
            $crate::PanickingTensor($b)
        )
    };
}

#[macro_export]
macro_rules! assert_relative_eq {
    ($a:expr, $b:expr) => {
        $crate::approx::assert_relative_eq!(
            $crate::PanickingTensor($a),
            $crate::PanickingTensor($b)
        )
    };
}

#[cfg(test)]
mod tests;
