use crate::plotting::{*};

#[derive(Clone)]
pub struct Line2d<T> {
    pub x_values: Vec<T>,
    pub y_values: Vec<T>,
    pub on_secondary_axis: bool,
    pub name: String,
    pub color: Option<RGBAColor>,
    pub color_map: Option<PlotBuilderColorMaps>,
    pub line_style: Option<LineStyle>,
    pub line_width: Option<u32>,
    pub marker_style: Option<MarkerStyle>,
    pub marker_fill: Option<MarkerFill>,
}

pub trait Line2dFunctions<T> {
    fn new(x_values: &Vec<T>, y_values: &Vec<T>) -> Self;
    fn new_y_only(y_values: &Vec<T>) -> Self;
    fn on_secondary_axis(&mut self, bool: bool) -> &mut Self;
    fn set_name(&mut self, title: &str) -> &mut Self;
    fn set_color(&mut self, color: RGBAColor) -> &mut Self;
    fn set_color_from_color_map(&mut self, normalized_value: f32, color_map: PlotBuilderColorMaps) -> &mut Self;
    fn set_color_map(&mut self, color_map: PlotBuilderColorMaps) -> &mut Self;
    fn set_line_style(&mut self, line_style: LineStyle) -> &mut Self;
    fn set_line_width(&mut self, line_width: u32) -> &mut Self;
    fn set_marker_style(&mut self, marker_style: MarkerStyle) -> &mut Self;
    fn set_marker_fill(&mut self, marker_style: MarkerFill) -> &mut Self;
}

macro_rules! impl_line_data_series_functions_per_type {
    ($T: ident) => {

        impl Default for Line2d<$T> {
            fn default() -> Line2d<$T> {
                Line2d::<$T> {
                    x_values: Vec::new(),
                    y_values: Vec::new(),
                    on_secondary_axis: false,
                    name: String::new(),
                    color: Option::None,
                    color_map: Option::None,
                    line_style: Option::None,
                    line_width: Option::None,
                    marker_style: Option::None,
                    marker_fill: Option::None,
                }
            }
        }

        impl Line2dFunctions<$T> for Line2d<$T> {
            fn new(x_values: &Vec<$T>, y_values: &Vec<$T>) -> Self {
                assert!(x_values.len() == y_values.len());
                Line2d::<$T> {
                    x_values: x_values.clone(),
                    y_values: y_values.clone(),
                    ..Default::default()
                }
            }

            fn new_y_only(y_values: &Vec<$T>) -> Self {
                let x_values: Vec<$T> = (0..y_values.len()).map(|x| x as $T).collect();
                Self::new(&x_values, y_values)
            }

            fn on_secondary_axis(&mut self, bool: bool) -> &mut Self {
                self.on_secondary_axis = bool;
                self
            }

            fn set_name(&mut self, title: &str) -> &mut Self {
                self.name = title.to_string();
                self
            }

            fn set_color(&mut self, color: RGBAColor) -> &mut Self {
                self.color = Option::Some(color);
                self
            }

            fn set_color_from_color_map(&mut self, normalized_value: f32, color_map: PlotBuilderColorMaps) -> &mut Self {
                self.color = Option::Some(
                    ColorMaps::get_color_from_map(vec![normalized_value], color_map)
                );
                self
            }

            fn set_color_map(&mut self, color_map: PlotBuilderColorMaps) -> &mut Self {
                self.color_map = Option::Some(color_map);
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

impl_line_data_series_functions_per_type!(i8);
impl_line_data_series_functions_per_type!(i16);
impl_line_data_series_functions_per_type!(i32);
impl_line_data_series_functions_per_type!(i64);

impl_line_data_series_functions_per_type!(isize);

impl_line_data_series_functions_per_type!(f32);
impl_line_data_series_functions_per_type!(f64);