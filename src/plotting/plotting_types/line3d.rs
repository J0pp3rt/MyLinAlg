use crate::plotting::{*};

#[derive(Clone)]
pub struct Line3d<T> {
    pub x_values: Vec<T>,
    pub y_values: Vec<T>,
    pub z_values: Vec<T>,
    pub name: String,
    pub color: Option<RGBAColor>,
    pub line_style: Option<LineStyle>,
    pub line_width: Option<u32>,
    pub marker_style: Option<MarkerStyle>,
    pub marker_fill: Option<MarkerFill>,
}

pub trait Line3dFunctions<T> {
    fn new(x_values: &Vec<T>, y_values: &Vec<T>, z_values: &Vec<T>) -> Self;
    fn set_name(&mut self, title: &str) -> &mut Self;
    fn set_color(&mut self, color: RGBAColor) -> &mut Self;
    fn set_line_style(&mut self, line_style: LineStyle) -> &mut Self;
    fn set_line_width(&mut self, line_width: u32) -> &mut Self;
    fn set_marker_style(&mut self, marker_style: MarkerStyle) -> &mut Self;
    fn set_marker_fill(&mut self, marker_style: MarkerFill) -> &mut Self;
}

macro_rules! impl_3d_line_functions {
    ($T: ident) => {
        impl Line3dFunctions<$T> for Line3d<$T> {
            fn new(x_values: &Vec<$T>, y_values: &Vec<$T>, z_values: &Vec<$T>) -> Self {
                assert!(x_values.len() == y_values.len());
                Self {
                    x_values: x_values.clone(),
                    y_values: y_values.clone(),
                    z_values: z_values.clone(),
                    name: String::new(),
                    color: Option::None,
                    line_style: Option::None,
                    line_width: Option::None,
                    marker_style: Option::None,
                    marker_fill: Option::None,
                }
            }

            fn set_name(&mut self, title: &str) -> &mut Self {
                self.name = title.to_string();
                self
            }

            fn set_color(&mut self, color: RGBAColor) -> &mut Self {
                self.color = Option::Some(color);
                self
            }

            fn set_line_style(&mut self, line_style: LineStyle) -> &mut Self{
                self.line_style = Option::Some(line_style);
                self
            }

            fn set_line_width(&mut self, line_width: u32) -> &mut Self{
                self.line_width = Option::Some(line_width);
                self
            }


            fn set_marker_style(&mut self, marker_style: MarkerStyle) -> &mut Self {
                self.marker_style = Option::Some(marker_style);
                self
            }

            fn set_marker_fill(&mut self, marker_fill: MarkerFill) -> &mut Self {
                self.marker_fill = Option::Some(marker_fill);
                self
            }
        }

    };
}

impl_3d_line_functions!(i8);
impl_3d_line_functions!(i16);
impl_3d_line_functions!(i32);
impl_3d_line_functions!(i64);

impl_3d_line_functions!(isize);

impl_3d_line_functions!(f32);
impl_3d_line_functions!(f64);