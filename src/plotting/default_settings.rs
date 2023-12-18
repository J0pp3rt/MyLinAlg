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

impl Clone for PlotStyleSettings {
    fn clone(&self) -> Self {
        Self { plot_width: self.plot_width.clone()
            , plot_height: self.plot_height.clone()
            , line_width: self.line_width.clone()

            , show_x_grid_minor: self.show_x_grid_minor.clone()
            , show_y_grid_minor: self.show_y_grid_minor.clone()
            , show_z_grid_minor: self.show_z_grid_minor.clone()
            , show_x_mesh: self.show_x_mesh.clone()
            , show_y_mesh: self.show_y_mesh.clone()
            , show_z_mesh: self.show_z_mesh.clone()
            , x_grid_minor_subdevisions: self.x_grid_minor_subdevisions.clone()
            , y_grid_minor_subdevisions: self.y_grid_minor_subdevisions.clone()
            , z_grid_minor_subdevisions: self.z_grid_minor_subdevisions.clone()
            , x_grid_major_subdevisions: self.x_grid_major_subdevisions.clone()
            , y_grid_major_subdevisions: self.y_grid_major_subdevisions.clone()
            , z_grid_major_subdevisions: self.z_grid_major_subdevisions.clone()
            , show_grid_major: self.show_grid_major.clone()
            , show_x_axis: self.show_x_axis.clone()
            , show_y_axis: self.show_y_axis.clone()
            , show_z_axis: self.show_z_axis.clone()
            , axis_equal_length: self.axis_equal_length.clone()
            , plot_3d_pitch: self.plot_3d_pitch.clone()
            , plot_3d_yaw: self.plot_3d_yaw.clone()
            , plot_3d_scale: self.plot_3d_scale.clone()
            , heatmap_n_points: self.heatmap_n_points.clone()
            , contour_n_points: self.contour_n_points.clone()
            , contour_n_lines: self.contour_n_lines.clone()
            , contour_band_width: self.contour_band_width.clone()
            , contour_alpha_value: self.contour_alpha_value.clone()
            , outer_figure_margins: self.outer_figure_margins.clone()
            , marker_fill: self.marker_fill.clone()
            , marker_style: self.marker_style.clone()
            , color_map_line: self.color_map_line.clone()
            , color_map_restricter_lower_bound: self.color_map_restricter_lower_bound.clone()
            , color_map_restricter_upper_bound: self.color_map_restricter_upper_bound.clone()
            , line_color: self.line_color.clone()
            , legend_show: self.legend_show.clone()
            , legend_location: self.legend_location.clone()
            , title: self.title.clone()
            , title_font_size: self.title_font_size.clone()
            , label_font_size: self.label_font_size.clone()
            , x_label_offset: self.x_label_offset.clone()
            , y_label_offset: self.y_label_offset.clone()
            , x_tick_mark_size: self.x_tick_mark_size.clone()
            , y_tick_mark_size: self.y_tick_mark_size.clone()
            , range_fitting_factor: self.range_fitting_factor.clone()
            , plotters_x_label_area_size: self.plotters_x_label_area_size.clone()
            , plotters_x_top_label_area_size: self.plotters_x_top_label_area_size.clone()
            , plotters_y_label_area_size: self.plotters_y_label_area_size.clone()
            , plotters_right_y_label_area_size: self.plotters_right_y_label_area_size.clone()
            , plotters_margin: self.plotters_margin.clone()
            , plotters_figure_padding: self.plotters_figure_padding.clone() 
            , plotters_legend_margin: self.plotters_legend_margin.clone()
            , plotters_legend_area_size: self.plotters_legend_area_size.clone()
            , plotters_legend_font_size: self.plotters_legend_font_size.clone()
            , plotters_legend_transparancy: self.plotters_legend_transparancy.clone()
            , plotters_legend_bar_size: self.plotters_legend_bar_size.clone()
            , plotters_legend_bar_shift_x: self.plotters_legend_bar_shift_x.clone()
            , plotters_legend_bar_shift_y: self.plotters_legend_bar_shift_y.clone()
        }
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
            contour_alpha_value: 0.5,
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

