use crate::{assert_almost_eq, EPS, REL_EPS};
use candle::{Device::Cpu, Tensor};

const SHAPE: (usize, usize, usize) = (100, 200, 300);

#[test]
fn abs_diff() {
    let a = Tensor::rand(0.0f32, 1.0, SHAPE, &Cpu).unwrap();
    let b = a
        .add(&a.rand_like(-0.99 * EPS, 0.99 * EPS).unwrap())
        .unwrap();
    assert_almost_eq!(a, b);
}

#[test]
#[should_panic]
fn abs_diff_panic() {
    let a = Tensor::rand(0.0f32, 1.0, SHAPE, &Cpu).unwrap();
    let b = (&a + 2.0 * EPS).unwrap();
    assert_almost_eq!(a, b);
}

#[test]
fn relative() {
    let a = Tensor::rand(0.0f32, 1.0, SHAPE, &Cpu).unwrap();
    let c = 1.0 + 0.99 * REL_EPS;
    let b = a.mul(&a.rand_like(1.0 / c, c).unwrap()).unwrap();
    assert_almost_eq!(a, b);
}

#[test]
#[should_panic]
fn relative_panic() {
    let a = Tensor::rand(0.0f32, 1.0, SHAPE, &Cpu).unwrap();
    let b = (&a * (1.0 + 2.0 * REL_EPS)).unwrap();
    assert_almost_eq!(a, b);
}
