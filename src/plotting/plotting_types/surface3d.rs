use std::marker::PhantomData;
use plotters::prelude::full_palette::GREY;

use crate::plotting::{*};

#[derive(Clone)]
pub struct SurfacePlot {}
#[derive(Clone)]
pub struct ContourPlot {}

#[derive(Clone)]
pub struct Surface3d<T, Type> {
    pub x_values: Option<Vec<T>>,
    pub y_values: Option<Vec<T>>,
    pub z_values: Option<Vec<Vec<T>>>,
    pub z_function: Option<Rc<Box<dyn Fn(Vec<f64>) -> f64>>>,
    pub n_contour_lines: Option<usize>,
    pub band_size: Option<f64>,
    pub color_map: PlotBuilderColorMaps,
    pub contour_alpha_factor: Option<f64>,
    pub plot_zeroth_contour: Option<bool>,
    pub plot_non_zeroth_contour: Option<bool>,
    pub plot_positve: Option<bool>,
    pub plot_zero: Option<bool>,
    pub plot_negative: Option<bool>,
    pub contour_force_plot_around_zero: Option<bool>,
    pub contour_force_plot_around_zero_range: Option<Range<f64>>,
    _type: PhantomData<Type>
}

impl<T> Default for Surface3d<T, SurfacePlot> {
    fn default() -> Surface3d<T, SurfacePlot> {
        Surface3d::<T, SurfacePlot> { 
            x_values: Option::None, 
            y_values: Option::None, 
            z_values: Option::None, 
            z_function: Option::None, 
            n_contour_lines: Option::None,
            band_size: Option::None,
            color_map: PlotBuilderColorMaps::ViridisInverse(1.),
            contour_alpha_factor: Option::None,
            plot_zeroth_contour: Option::None,
            plot_non_zeroth_contour: Option::None,
            plot_positve: Option::None,
            plot_zero: Option::None,
            plot_negative: Option::None,
            contour_force_plot_around_zero: Option::None,
            contour_force_plot_around_zero_range: Option::None,
            _type: PhantomData::<SurfacePlot>
        }
    }
}

impl<T> Default for Surface3d<T, ContourPlot> {
    fn default() -> Surface3d<T, ContourPlot> {
        Surface3d::<T, ContourPlot> { 
            x_values: Option::None, 
            y_values: Option::None, 
            z_values: Option::None, 
            z_function: Option::None, 
            n_contour_lines: Option::None,
            band_size: Option::None,
            color_map: PlotBuilderColorMaps::ConstantGray(0.5),
            contour_alpha_factor: Option::None,
            plot_zeroth_contour: Option::None,
            plot_non_zeroth_contour: Option::None,
            plot_positve: Option::None,
            plot_zero: Option::None,
            plot_negative: Option::None,
            contour_force_plot_around_zero: Option::None,
            contour_force_plot_around_zero_range: Option::None,
            _type: PhantomData::<ContourPlot>
        }
    }
}

pub trait Surface3dFunctions<T> {
    fn new_surface_fn(z_function: Rc<Box<dyn Fn(Vec<f64>) -> f64>>) -> Surface3d<T, SurfacePlot>;
    fn new_surface_xyz(x_values: &Vec<T>, y_values: &Vec<T>, z_values: &Vec<Vec<T>>) -> Surface3d<T, SurfacePlot>;
    fn set_colormap(&mut self, color_map: PlotBuilderColorMaps) -> &Self;
    fn get_color(&self, input_variables: Vec<f32>) -> RGBAColor;
}

pub trait ContourFunctions<T> {
    fn new_contour_fn(z_function: Rc<Box<dyn Fn(Vec<f64>) -> f64>>) -> Surface3d<T, ContourPlot>;
    fn new_contour_xyz(x_values: &Vec<T>, y_values: &Vec<T>, z_values: &Vec<Vec<T>>) -> Surface3d<T, ContourPlot>;
    fn set_colormap(&mut self, color_map: PlotBuilderColorMaps) -> &Self;
    fn get_color(&self, input_variables: Vec<f32>) -> RGBAColor;
    fn use_constraint_contour_preset(&mut self) -> &mut Self;
    fn use_constraint_filled_preset(&mut self) -> &mut Self;
}

macro_rules! impl_3d_line_functions {
    ($T: ident) => {
        impl Surface3dFunctions<$T> for Surface3d<$T, SurfacePlot> {
            fn new_surface_fn(z_function: Rc<Box<dyn Fn(Vec<f64>) -> f64>>) -> Surface3d<$T, SurfacePlot> {
                Surface3d::<$T, SurfacePlot> {
                    z_function: Option::Some(z_function),
                    .. Default::default()
                }
            }

            fn new_surface_xyz(x_values: &Vec<$T>, y_values: &Vec<$T>, z_values: &Vec<Vec<$T>>) -> Surface3d<$T, SurfacePlot> {
                Surface3d::<$T, SurfacePlot> {
                    x_values: Option::Some(x_values.clone()),
                    y_values: Option::Some(y_values.clone()),
                    z_values: Option::Some(z_values.clone()),
                    .. Default::default()
                }
            }

            fn set_colormap(&mut self, color_map: PlotBuilderColorMaps) -> &Self {
                match color_map {
                    PlotBuilderColorMaps::Palette99(_) => {println!("The palette colormap and Surface plots are a recipe for disaster, making no change")},
                    PlotBuilderColorMaps::Palette99ReversedOrder(_) => {println!("The palette colormap and Surface plots are a recipe for disaster, making no change")}
                    _ => {self.color_map = color_map;} 
                }
                self
            }
            fn get_color(&self, input_variables: Vec<f32>) -> RGBAColor {
                ColorMaps::get_color_from_map(input_variables, self.color_map.clone())
            }
        }

        impl ContourFunctions<$T> for Surface3d<$T, ContourPlot> {
            fn new_contour_fn(z_function: Rc<Box<dyn Fn(Vec<f64>) -> f64>>) -> Surface3d<$T, ContourPlot> {
                Surface3d::<$T, ContourPlot> {
                    z_function: Option::Some(z_function),
                    .. Default::default()
                }
            }

            fn new_contour_xyz(x_values: &Vec<$T>, y_values: &Vec<$T>, z_values: &Vec<Vec<$T>>) -> Surface3d<$T, ContourPlot> {
                Surface3d::<$T, ContourPlot> {
                    x_values: Option::Some(x_values.clone()),
                    y_values: Option::Some(y_values.clone()),
                    z_values: Option::Some(z_values.clone()),
                    .. Default::default()
                }
            }

            fn set_colormap(&mut self, color_map: PlotBuilderColorMaps) -> &Self {
                match color_map {
                    PlotBuilderColorMaps::Palette99(_) => {println!("The palette colormap and contour plots are a recipe for disaster, making no change (note to self: might be possible if contour line number is index (devision of...))")},
                    PlotBuilderColorMaps::Palette99ReversedOrder(_) => {println!("The palette colormap and Surface plots are a recipe for disaster, making no change (note to self: might be possible if contour line number is index (devision of...))")}
                    _ => {self.color_map = color_map;} 
                }
                self
            }
            fn get_color(&self, input_variables: Vec<f32>) -> RGBAColor {
                match self.contour_alpha_factor {
                    Option::Some(factor) => {
                        let mut color = ColorMaps::get_color_from_map(input_variables, self.color_map.clone());
                        color.3 = color.3 * factor;
                        color
                    },
                    Option::None => {
                        ColorMaps::get_color_from_map(input_variables, self.color_map.clone())
                    }
                }
                
            }

            fn use_constraint_contour_preset(&mut self) -> &mut Self {
                self.plot_zeroth_contour = Option::Some(false);
                self.plot_non_zeroth_contour = Option::Some(false);
                self.color_map = PlotBuilderColorMaps::ConstantRed(1.);
                self.contour_force_plot_around_zero = Option::Some(true);
                self.contour_force_plot_around_zero_range = Option::Some(0. .. 0.01);

                self.plot_negative = Option::Some(false);
                self.contour_alpha_factor = Option::Some(0.5);


                self
            }

            fn use_constraint_filled_preset(&mut self) -> &mut Self {
                self.plot_zeroth_contour = Option::Some(false);
                self.plot_non_zeroth_contour = Option::Some(true);
                self.color_map = PlotBuilderColorMaps::ConstantRed(1.);

                self.plot_negative = Option::Some(false);
                self.band_size = Option::Some(1.);

                self.contour_alpha_factor = Option::Some(0.2);

                self
            }

        }
    }
}

impl_3d_line_functions!(i8);
impl_3d_line_functions!(i16);
impl_3d_line_functions!(i32);
impl_3d_line_functions!(i64);

impl_3d_line_functions!(isize);

impl_3d_line_functions!(f32);
impl_3d_line_functions!(f64);