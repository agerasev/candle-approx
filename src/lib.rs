use candle_core::{Result, Tensor};

pub const DEFAULT_EPSILON: f64 = 1e-4;
pub const DEFAULT_MAX_RELATIVE: f64 = 1e-4;

pub fn abs_diff_eq(a: &Tensor, b: &Tensor, epsilon: f64) -> Result<Tensor> {
    a.sub(b)?.abs()?.le(epsilon)
}

pub fn relative_eq(a: &Tensor, b: &Tensor, epsilon: f64, max_relative: f64) -> Result<Tensor> {
    let norm2 = a.abs()?.add(&b.abs()?)?;
    let diff = a.sub(b)?.abs()?;
    diff.le(&(norm2 * (0.5 * max_relative))?.maximum(epsilon)?)
}

pub fn all(a: &Tensor) -> Result<bool> {
    Ok(a.flatten_all()?.min(0)?.to_scalar::<u8>()? != 0)
}

#[macro_export]
macro_rules! assert_abs_diff_eq {
    ($a:expr, $b:expr, $eps:expr $(,)?) => {{
        let (a, b) = (&$a, &$b);
        let eps = $eps;
        assert!(
            $crate::all(&$crate::abs_diff_eq(a, b, eps).unwrap()).unwrap(),
            "assert_abs_diff_eq failed (epsilon = {:.0e}):\n{}:\n{}\n{}:\n{}",
            eps,
            stringify!($a),
            a,
            stringify!($b),
            b,
        );
    }};
    ($a:expr, $b:expr $(,)?) => {{
        assert_abs_diff_eq!($a, $b, $crate::DEFAULT_EPSILON);
    }};
}

#[macro_export]
macro_rules! assert_relative_eq {
    ($a:expr, $b:expr, $eps:expr, $max_rel:expr $(,)?) => {{
        let (a, b) = (&$a, &$b);
        let (eps, max_rel) = ($eps, $max_rel);
        assert!(
            $crate::all(&$crate::relative_eq(a, b, eps, max_rel).unwrap()).unwrap(),
            "assert_relative_eq failed (epsilon = {:.0e}, max_relative = {:.0e}):\n{}:\n{}\n{}:\n{}",
            eps,
            max_rel,
            stringify!($a),
            a,
            stringify!($b),
            b,
        );
    }};
    ($a:expr, $b:expr $(,)?) => {{
        assert_relative_eq!(
            $a,
            $b,
            $crate::DEFAULT_EPSILON,
            $crate::DEFAULT_MAX_RELATIVE,
        );
    }};
}

#[cfg(test)]
mod tests;
