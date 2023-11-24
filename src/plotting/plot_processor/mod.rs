use std::marker::PhantomData;

use crate::plotting::{*};

pub mod plotters_backend;
pub use plotters_backend::{*};

pub struct NoPlotBackend {}
pub struct PlotPyBackend {}
pub struct  PlottersBackend {}

// pub trait 

pub struct PlotProcessor<T, Backend> {
    plots: PlotBuilder<T>,
    backend: PhantomData<Backend>,
}

impl<T> PlotProcessor<T, NoPlotBackend> {
    pub fn new_unitialized(plots: PlotBuilder<T>) -> Self {
        Self {plots, backend:PhantomData::<NoPlotBackend>}
    }

    pub fn use_plotters_backend(self) -> PlotProcessor<T, PlottersBackend> {
        PlotProcessor { plots: self.plots, backend: PhantomData::<PlottersBackend>}
    }

    pub fn use_plotpy_backend(self) -> PlotProcessor<T, PlotPyBackend> {
        PlotProcessor { plots: self.plots, backend: PhantomData::<PlotPyBackend>}
    }
}
impl<T> PlotProcessor<T, PlottersBackend> {
    pub fn new_plotters_backend(plots: PlotBuilder<T>) -> Self {
        Self {plots, backend:PhantomData::<PlottersBackend>}
    }
}

impl<T> PlotProcessor<T, PlotPyBackend> {
    pub fn new_plotpy_backend(plots: PlotBuilder<T>) -> Self {
        Self {plots, backend:PhantomData::<PlotPyBackend>}
    }
}