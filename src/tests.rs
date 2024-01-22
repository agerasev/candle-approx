use crate::{assert_abs_diff_eq, assert_relative_eq};
use candle_core::{Device::Cpu, Tensor};

const EPS: f64 = 0.98 * crate::DEFAULT_EPSILON;
const REL_EPS: f64 = 0.98 * crate::DEFAULT_MAX_RELATIVE;
const SHAPE: (usize, usize, usize) = (20, 30, 40);

#[test]
fn abs_diff() {
    let a = Tensor::randn(0.0f32, 1.0, SHAPE, &Cpu).unwrap();
    let noise = a.rand_like(-EPS, EPS).unwrap();
    let b = a.add(&noise).unwrap();
    assert_abs_diff_eq!(a, b);
}

#[test]
#[should_panic]
fn abs_diff_panic() {
    let a = Tensor::randn(0.0f32, 1.0, SHAPE, &Cpu).unwrap();
    let b = (&a + 2.0 * EPS).unwrap();
    assert_abs_diff_eq!(a, b);
}

#[test]
fn relative() {
    let a = Tensor::randn(0.0f32, 1.0, SHAPE, &Cpu).unwrap();
    let factor = 1.0 + REL_EPS;
    let b = a.mul(&a.rand_like(1.0 / factor, factor).unwrap()).unwrap();
    assert_relative_eq!(a, b);
}

#[test]
#[should_panic]
fn relative_panic() {
    let a = Tensor::randn(0.0f32, 1.0, SHAPE, &Cpu).unwrap();
    let b = (&a * (1.0 + 2.0 * REL_EPS)).unwrap();
    assert_relative_eq!(a, b);
}
