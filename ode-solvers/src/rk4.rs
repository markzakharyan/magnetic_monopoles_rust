//! Explicit Runge-Kutta method of order 4 with fixed step size.

use crate::dop_shared::{IntegrationError, Stats, System};

use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, Scalar, Matrix4};
use num_traits::Zero;
use simba::scalar::{ClosedAdd, ClosedMul, ClosedNeg, ClosedSub, SubsetOf};
use std::boxed::Box;

/// Structure containing the parameters for the numerical integration.
pub struct Rk4<V, F>
where
    F: System<V>,
{
    f: F,
    x: f64,
    y: V,
    x_end: f64,
    step_size: f64,
    half_step: f64,
    x_out: Vec<f64>,
    y_out: Vec<V>,
    stats: Stats,
    event_fn: Option<Box<dyn Fn(f64, &V, &Matrix4<f64>) -> bool>>,
    boost_back: Matrix4<f64>,
}

impl<T, D: Dim, F> Rk4<OVector<T, D>, F>
where
    f64: From<T>,
    T: Copy + SubsetOf<f64> + Scalar + ClosedAdd + ClosedMul + ClosedSub + ClosedNeg + Zero,
    F: System<OVector<T, D>>,
    OVector<T, D>: std::ops::Mul<f64, Output = OVector<T, D>>,
    DefaultAllocator: Allocator<T, D>,
{
    /// Default initializer for the structure
    ///
    /// # Arguments
    ///
    /// * `f`           - Structure implementing the System<V> trait
    /// * `x`           - Initial value of the independent variable (usually time)
    /// * `y`           - Initial value of the dependent variable(s)
    /// * `x_end`       - Final value of the independent variable
    /// * `step_size`   - Step size used in the method
    ///
    pub fn new(f: F, x: f64, y: OVector<T, D>, x_end: f64, step_size: f64, boost_back: Matrix4<f64>) -> Self {
        Rk4 {
            f,
            x,
            y,
            x_end,
            step_size,
            half_step: step_size / 2.,
            x_out: Vec::new(),
            y_out: Vec::new(),
            stats: Stats::new(),
            event_fn: None,
            boost_back: boost_back,
        }
    }

    // Add a method to set the event function
    pub fn with_event_fn(mut self, event_fn: Box<dyn Fn(f64, &OVector<T, D>, &Matrix4<f64>) -> bool>) -> Self {
        self.event_fn = Some(event_fn);
        self
    }

    /// Core integration method.
    pub fn integrate(&mut self) -> Result<Stats, IntegrationError> {
        // Save initial values
        self.x_out.push(self.x);
        self.y_out.push(self.y.clone());

        let num_steps = ((self.x_end - self.x) / self.step_size).ceil() as usize;
        for _ in 0..num_steps {
            let (x_new, y_new) = self.step();

            // Check if the event function is set and if the event has occurred
            if let Some(ref event_fn) = self.event_fn {
                if event_fn(x_new, &y_new, &self.boost_back) {
                    break; // Stop integration if the event occurs
                }
            }

            self.x_out.push(x_new);
            self.y_out.push(y_new.clone());

            self.x = x_new;
            self.y = y_new;

            self.stats.num_eval += 4;
            self.stats.accepted_steps += 1;
        }
        Ok(self.stats)
    }

    /// Performs one step of the Runge-Kutta 4 method.
    fn step(&self) -> (f64, OVector<T, D>) {
        let (rows, cols) = self.y.shape_generic();
        let mut k = vec![OVector::zeros_generic(rows, cols); 12];

        self.f.system(self.x, &self.y, &mut k[0]);
        self.f.system(
            self.x + self.half_step,
            &(self.y.clone() + k[0].clone() * self.half_step),
            &mut k[1],
        );
        self.f.system(
            self.x + self.half_step,
            &(self.y.clone() + k[1].clone() * self.half_step),
            &mut k[2],
        );
        self.f.system(
            self.x + self.step_size,
            &(self.y.clone() + k[2].clone() * self.step_size),
            &mut k[3],
        );

        let x_new = self.x + self.step_size;
        let y_new = &self.y
            + (k[0].clone() + k[1].clone() * 2.0 + k[2].clone() * 2.0 + k[3].clone())
                * (self.step_size / 6.0);
        (x_new, y_new)
    }

    /// Getter for the independent variable's output.
    pub fn x_out(&self) -> &Vec<f64> {
        &self.x_out
    }

    /// Getter for the dependent variables' output.
    pub fn y_out(&self) -> &Vec<OVector<T, D>> {
        &self.y_out
    }
}