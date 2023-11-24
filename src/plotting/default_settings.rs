use std::f64::consts::PI;

use crate::plotting::{*};

pub struct PlotStyleSettings {
    pub plot_width: u32,
    pub plot_height: u32,
    pub line_width: u32,
    pub show_x_mesh: bool,
    pub show_y_mesh: bool,
    pub show_z_mesh: bool,
    pub show_x_grid_minor: bool,
    pub show_y_grid_minor: bool,
    pub show_z_grid_minor: bool,
    pub x_grid_minor_subdevisions: Option<usize>,
    pub y_grid_minor_subdevisions: Option<usize>,
    pub z_grid_minor_subdevisions: Option<usize>,
    pub x_grid_major_subdevisions: Option<usize>,
    pub y_grid_major_subdevisions: Option<usize>,
    pub z_grid_major_subdevisions: Option<usize>,
    pub show_x_grid_major: bool,
    pub show_y_grid_major: bool,
    pub show_z_grid_major: bool,
    pub show_x_axis: bool,
    pub show_y_axis: bool,
    pub show_z_axis: bool,
    pub plot_3d_pitch: f64,
    pub plot_3d_yaw: f64,
    pub plot_3d_scale: f64,
    pub outer_figure_margins: u32,
    pub marker_fill: MarkerFill,
    pub marker_style: MarkerStyle,
    pub color_map: Box<dyn ColorMap<RGBAColor>>,
    pub line_color: Option<RGBAColor>,
    pub title: String,
    pub plotters_x_label_area_size: u32,
    pub plotters_y_label_area_size: u32,
    pub plotters_right_y_label_area_size: u32,
    pub plotters_margin: u32,
    pub plotters_figure_padding: u32,

}

impl PlotStyleSettings {
    pub fn new() -> Self {
        Self { ..Default::default()}
    }
}

impl Default for PlotStyleSettings {
    fn default() -> Self {
        PlotStyleSettings { 
            plot_width: 500,
            plot_height: 500,
            line_width: 2,
            show_x_mesh: true,
            show_y_mesh: true,
            show_z_mesh: true,
            show_x_axis: true,
            show_y_axis: true,
            show_z_axis: true,
            plot_3d_pitch: -0.8 / 2. * PI,
            plot_3d_yaw: 0.05,
            plot_3d_scale: 0.7,
            show_x_grid_major: false,
            show_y_grid_major: false,
            show_z_grid_major: false,
            show_x_grid_minor: false,
            show_y_grid_minor: false,
            show_z_grid_minor: false,
            x_grid_minor_subdevisions: Option::None,
            y_grid_minor_subdevisions: Option::None,
            z_grid_minor_subdevisions: Option::None,
            x_grid_major_subdevisions: Option::None,
            y_grid_major_subdevisions: Option::None,
            z_grid_major_subdevisions: Option::None,
            outer_figure_margins: 10,
            marker_fill: MarkerFill::Filled,
            marker_style: MarkerStyle::None,
            color_map: Box::new(ViridisRGBA {}),
            line_color: Option::None,
            title: String::new(),
            plotters_x_label_area_size: 35,
            plotters_y_label_area_size: 50,
            plotters_right_y_label_area_size: 10,
            plotters_margin: 5,
            plotters_figure_padding: 0,
            }
    }
}