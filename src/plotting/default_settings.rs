use std::f64::consts::PI;

use crate::plotting::{*};

#[derive(Clone)]
pub enum LegendPosition {
    NorthEast,
    North,
    NorthWest,
    West,
    SouthWest,
    South,
    SouthEast,
    East
}

#[derive(Clone)]
pub struct PlotStyleSettings {
    pub plot_width: u32,
    pub plot_height: u32,
    pub line_width: u32,
    pub show_x_grid_minor: bool,
    pub show_y_grid_minor: bool,
    pub show_z_grid_minor: bool,
    pub show_x_mesh: bool,
    pub show_y_mesh: bool,
    pub show_z_mesh: bool,
    pub x_grid_minor_subdevisions: Option<usize>,
    pub y_grid_minor_subdevisions: Option<usize>,
    pub z_grid_minor_subdevisions: Option<usize>,
    pub x_grid_major_subdevisions: Option<usize>,
    pub y_grid_major_subdevisions: Option<usize>,
    pub z_grid_major_subdevisions: Option<usize>,
    pub show_grid_major: bool,
    pub show_x_axis: bool,
    pub show_y_axis: bool,
    pub show_z_axis: bool,
    pub axis_equal_length: bool,
    pub plot_3d_pitch: f64,
    pub plot_3d_yaw: f64,
    pub plot_3d_scale: f64,
    pub heatmap_n_points: usize,
    pub contour_n_points: usize,
    pub contour_n_lines: usize,
    pub contour_band_width: f32,
    pub contour_alpha_value: f64,
    pub polygon_line_width: u32,
    pub polygon_filled: bool,
    pub outer_figure_margins: u32,
    pub marker_fill: MarkerFill,
    pub marker_style: MarkerStyle,
    pub color_map_line: PlotBuilderColorMaps,
    pub color_map_restricter_lower_bound: f32,
    pub color_map_restricter_upper_bound: f32,
    pub line_color: Option<RGBAColor>,
    pub legend_show: bool,
    pub legend_location: LegendPosition,
    pub title: String,
    pub title_font_size: u32,
    pub label_font_size: u32,
    pub x_label_offset: i32,
    pub y_label_offset: i32,
    pub x_tick_mark_size: i32,
    pub y_tick_mark_size: i32,
    pub range_fitting_factor: f64,
    pub plotters_x_label_area_size: i32,
    pub plotters_x_top_label_area_size: i32,
    pub plotters_y_label_area_size: i32,
    pub plotters_right_y_label_area_size: i32,
    pub plotters_margin: i32,
    pub plotters_figure_padding: i32,
    pub plotters_legend_margin: i32,
    pub plotters_legend_area_size: i32,
    pub plotters_legend_font_size: i32,
    pub plotters_legend_transparancy: f64,
    pub plotters_legend_bar_size: i32,
    pub plotters_legend_bar_shift_x: i32,
    pub plotters_legend_bar_shift_y: i32,



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
            line_width: 3,
            show_x_axis: true,
            show_y_axis: true,
            show_z_axis: true,
            axis_equal_length: false,
            plot_3d_pitch: -0.8 / 2. * PI,
            plot_3d_yaw: 0.05,
            plot_3d_scale: 0.7,
            heatmap_n_points: 150,
            contour_n_points: 1000,
            contour_n_lines: 20,
            contour_band_width: 0.08,
            contour_alpha_value: 0.5, // alpha what?
            polygon_line_width: 10,
            polygon_filled: true,
            show_grid_major: false,
            show_x_grid_minor: false,
            show_y_grid_minor: false,
            show_z_grid_minor: false,
            show_x_mesh: true,
            show_y_mesh: true,
            show_z_mesh: true,
            x_grid_minor_subdevisions: Option::None,
            y_grid_minor_subdevisions: Option::None,
            z_grid_minor_subdevisions: Option::None,
            y_grid_major_subdevisions: Option::Some(10),
            x_grid_major_subdevisions: Option::Some(10),
            z_grid_major_subdevisions: Option::None,
            outer_figure_margins: 10,
            marker_fill: MarkerFill::Filled,
            marker_style: MarkerStyle::None,
            color_map_line: PlotBuilderColorMaps::Palette99,
            color_map_restricter_lower_bound: 0.25,
            color_map_restricter_upper_bound: 1.,
            line_color: Option::None,
            legend_show: false,
            legend_location: LegendPosition::NorthEast,
            title: String::new(),
            title_font_size: 40,
            label_font_size: 20,
            x_label_offset: 0,
            y_label_offset: 0,
            x_tick_mark_size: 13,
            y_tick_mark_size: 5,
            range_fitting_factor: 0.05,
            plotters_x_label_area_size: 55,
            plotters_x_top_label_area_size: 0,
            plotters_y_label_area_size: 60,
            plotters_right_y_label_area_size: 10,
            plotters_margin: 25,
            plotters_figure_padding: 0,
            plotters_legend_margin: 15,
            plotters_legend_area_size: 5,
            plotters_legend_font_size: 15,
            plotters_legend_transparancy: 0.2,
            plotters_legend_bar_size: 4,
            plotters_legend_bar_shift_x: 10,
            plotters_legend_bar_shift_y: 10,
            }
    }
}

