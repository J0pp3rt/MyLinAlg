use crate::plotting::{*};

#[derive(Clone)]
pub struct Surface3d<T> {
    pub x_values: Vec<T>,
    pub y_values: Vec<T>,
    pub z_values: Vec<Vec<T>>,
    pub line_style: Option<LineStyle>,
}

pub trait Surface3dFunctions<T> {
    fn new(x_values: &Vec<T>, y_values: &Vec<T>, z_values: &Vec<Vec<T>>) -> Self;
    fn set_line_style(&mut self, line_style: LineStyle) -> &mut Self;
}

macro_rules! impl_3d_line_functions {
    ($T: ident) => {
        impl Surface3dFunctions<$T> for Surface3d<$T> {
            fn new(x_values: &Vec<$T>, y_values: &Vec<$T>, z_values: &Vec<Vec<$T>>) -> Self {
                Self {
                    x_values: x_values.clone(),
                    y_values: y_values.clone(),
                    z_values: z_values.clone(),
                    line_style: Option::None,
                }
            }

            fn set_line_style(&mut self, line_style: LineStyle) -> &mut Self {
                self.line_style = Option::Some(line_style);

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