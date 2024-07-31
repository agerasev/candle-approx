pub const EPS: f64 = 1e-4;
pub const REL_EPS: f64 = 1e-4;

#[macro_export]
macro_rules! assert_almost_eq {
    ($a:expr, $b:expr, $eps:expr, $rel_eps:expr $(,)?) => {{
        let (a, b) = ($a, $b);
        let (eps, rel_eps) = ($eps as f64, $rel_eps as f64);
        let norm = a.abs().unwrap().add(&b.abs().unwrap()).unwrap();
        let diff = a.sub(&b).unwrap().abs().unwrap();
        let mask = diff
            .gt(&(norm * rel_eps).unwrap().maximum(eps).unwrap())
            .unwrap();
        assert!(
            mask.flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<u8>()
                .unwrap()
                == 0,
            "Tensors are not equal (eps = {:.0e}, relative eps = {:.0e}):\n{}\n{}",
            eps,
            rel_eps,
            a,
            b,
        );
    }};
    ($a:expr, $b:expr $(,)?) => {{
        assert_almost_eq!($a, $b, $crate::EPS, $crate::REL_EPS);
    }};
}

#[cfg(test)]
mod tests;
