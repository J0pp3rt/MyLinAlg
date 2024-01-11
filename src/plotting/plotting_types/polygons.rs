use crate::plotting::{*};

#[derive(Clone)]
pub struct PolygonElement<T> {
    pub x_values: Vec<T>,
    pub y_values: Vec<T>,
    pub z_value: T,
    pub color: Option<RGBAColor>,
    pub line_width: Option<u32>,
    pub filled: Option<bool>,
}

#[derive(Clone)]
pub struct PolygonSet<T> {
    pub polygons: Vec<PolygonElement<T>>,
    pub color: Option<RGBAColor>,
    pub color_map: PlotBuilderColorMaps,
    pub line_width: Option<u32>,
    pub filled: Option<bool>,
}

pub trait PolygonFunction<T> {
    fn new(x_values: Vec<T>, y_values: Vec<T>) -> Self;
    fn new_with_z(x_values: Vec<T>, y_values: Vec<T>, z_value: T) -> Self;
    fn set_z_value(&mut self, z_value: T) -> &mut Self;
    fn set_color(&mut self, color: RGBAColor) -> &mut Self;
    fn set_color_from_color_map(&mut self, normalized_value: f32, color_map: PlotBuilderColorMaps) -> &mut Self;
    fn set_line_width(&mut self, line_width: u32) -> &mut Self;
    fn set_fill(&mut self, is_filled: bool) -> *mut Self;
}

pub trait PolygonSetFunction<T> {
    fn new() -> Self;
    fn add_polygon(&mut self, polygon: PolygonElement<T>) -> &mut Self;
    fn add_simple_polygon(&mut self, x_values: Vec<T>, y_values: Vec<T>) -> &mut Self;
    fn add_simple_polygon_with_z(&mut self, x_values: Vec<T>, y_values: Vec<T>, z_value: T) -> &mut Self;
    fn set_color_map(&mut self, color_map: PlotBuilderColorMaps) -> &mut Self;
    fn set_line_width(&mut self, line_width: u32) -> &mut Self;
    fn set_fill(&mut self, is_filled: bool) -> *mut Self;
}

macro_rules! impl_triangles_functions_per_type {
    ($T: ident) => {

        impl Default for PolygonElement<$T> {
            fn default() -> PolygonElement<$T> {
                PolygonElement::<$T> {
                    x_values: Vec::new(),
                    y_values: Vec::new(),
                    z_value: 0 as $T,
                    color: Option::None,
                    line_width: Option::None,
                    filled: Option::None,
                }
            }
        }

        impl Default for PolygonSet<$T> {
            fn default() -> PolygonSet<$T> {
                PolygonSet::<$T> {
                    polygons: Vec::new(),
                    color: Option::None,
                    color_map: PlotBuilderColorMaps::Vulcano(1.),
                    line_width: Option::None,
                    filled: Option::None,
                }
            }
        }

        impl PolygonFunction<$T> for PolygonElement<$T> {
            fn new(x_values: Vec<$T>, y_values: Vec<$T>) -> Self {
                assert!(x_values.len() == y_values.len());
                PolygonElement::<$T> {
                    x_values: x_values.clone(),
                    y_values: y_values.clone(),
                    ..Default::default()
                }
            }

            fn new_with_z(x_values: Vec<$T>, y_values: Vec<$T>, z_value: $T) -> Self {
                assert!(x_values.len() == y_values.len());
                PolygonElement::<$T> {
                    x_values: x_values.clone(),
                    y_values: y_values.clone(),
                    z_value: z_value,
                    ..Default::default()
                }
            }

            fn set_z_value(&mut self, z_value: $T) -> &mut Self {
                self.z_value = z_value;

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

            fn set_line_width(&mut self, line_width: u32) -> &mut Self{
                self.line_width = Option::Some(line_width);
                self
            }

            fn set_fill(&mut self, is_filled: bool) -> *mut Self {
                self.filled = Option::Some(is_filled);

                self
            }
        }

        impl PolygonSetFunction<$T> for PolygonSet<$T> {
            fn new() -> Self {
                PolygonSet::<$T> {
                    ..Default::default()
                }
            }

            fn add_polygon(&mut self, polygon: PolygonElement<$T>) -> &mut Self {
                self.polygons.push(polygon);

                self
            }

            fn add_simple_polygon(&mut self, x_values: Vec<$T>, y_values: Vec<$T>) -> &mut Self {
                let polygon = PolygonElement::new(x_values, y_values);
                self.add_polygon(polygon)
            }

            fn add_simple_polygon_with_z(&mut self, x_values: Vec<$T>, y_values: Vec<$T>, z_value: $T) -> &mut Self {
                let mut polygon = PolygonElement::new(x_values, y_values);
                polygon.z_value = z_value;
                self.add_polygon(polygon)
            }

            fn set_color_map(&mut self, color_map: PlotBuilderColorMaps) -> &mut Self {
                self.color_map = color_map;
                self
            }

            fn set_line_width(&mut self, line_width: u32) -> &mut Self{
                self.line_width = Option::Some(line_width);
                self
            }

            fn set_fill(&mut self, is_filled: bool) -> *mut Self {
                self.filled = Option::Some(is_filled);

                self
            }
        }
    }
}

impl_triangles_functions_per_type!(i8);
impl_triangles_functions_per_type!(i16);
impl_triangles_functions_per_type!(i32);
impl_triangles_functions_per_type!(i64);

impl_triangles_functions_per_type!(isize);

impl_triangles_functions_per_type!(f32);
impl_triangles_functions_per_type!(f64);