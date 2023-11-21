use std::{marker::PhantomData, ops::RangeBounds};
use plotters::coord::types::RangedCoordf64;
use plotters::coord::Shift;

use crate::{*};

const STD_PLOT_WIDTH: u32 = 1000;
const STD_PLOT_HEIGHT: u32 = 1000;

pub struct PlotStyleSettings {
    line_width: u32,
    show_x_mesh: bool,
    show_y_mesh: bool,
    show_z_mesh: bool,
    outer_figure_margins: u32,
    marker_fill: MarkerFill,
    marker_style: MarkerStyle,
    color_map: Box<dyn ColorMap<RGBAColor>>,
    line_color: Option<RGBAColor>,
    title: String,
    plotters_x_label_area_size: u32,
    plotters_y_label_area_size: u32,
    plotters_right_y_label_area_size: u32,
    plotters_margin: u32,
    plotters_figure_padding: u32,

}

impl PlotStyleSettings {
    pub fn new() -> Self {
        Self { ..Default::default()}
    }
}

impl Default for PlotStyleSettings {
    fn default() -> Self {
        PlotStyleSettings { 
            line_width: 2,
            show_x_mesh: true,
            show_y_mesh: true,
            show_z_mesh: true,
            outer_figure_margins: 10,
            marker_fill: MarkerFill::Filled,
            marker_style: MarkerStyle::None,
            color_map: Box::new(ViridisRGBA {}),
            line_color: Option::None,
            title: String::new(),
            plotters_x_label_area_size: 35,
            plotters_y_label_area_size: 40,
            plotters_right_y_label_area_size: 40,
            plotters_margin: 5,
            plotters_figure_padding: 0,
            }
    }
}

#[derive(Clone)]
pub enum MarkerStyle {
    None,
    Circle(u32),
    Diamand(u32),
    Cross(u32),
}

#[derive(Clone)]
pub enum MarkerFill {
    Filled,
    NotFilled
}

#[derive(Clone)]
pub enum LineStyle {
    Solid,
    Dashed,
}

#[derive(Clone)]
pub struct Line2d<T> {
    x_values: Vec<T>,
    y_values: Vec<T>,
    on_secondary_axis: bool,
    name: Option<String>,
    color: Option<RGBAColor>,
    line_style: Option<LineStyle>,
    line_width: Option<u32>,
    marker_style: Option<MarkerStyle>,
    marker_fill: Option<MarkerFill>,
}

pub trait Line2dFunctions<T> {
    fn new(x_values: &Vec<T>, y_values: &Vec<T>) -> Self;
    fn new_y_only(y_values: &Vec<T>) -> Self;
    fn on_secondary_axis(&mut self, bool: bool) -> &mut Self;
    fn set_name(&mut self, title: &str) -> &mut Self;
    fn set_color(&mut self, color: RGBAColor) -> &mut Self;
    fn set_line_style(&mut self, line_style: LineStyle) -> &mut Self;
    fn set_line_width(&mut self, line_width: u32) -> &mut Self;
    fn set_marker_style(&mut self, marker_style: MarkerStyle) -> &mut Self;
    fn set_marker_fill(&mut self, marker_style: MarkerFill) -> &mut Self;
}

macro_rules! impl_line_data_series_functions_per_type {
    ($T: ident) => {
        impl Line2dFunctions<$T> for Line2d<$T> {
            fn new(x_values: &Vec<$T>, y_values: &Vec<$T>) -> Self {
                assert!(x_values.len() == y_values.len());
                Self {
                    x_values: x_values.clone(),
                    y_values: y_values.clone(),
                    on_secondary_axis: false,
                    name: Option::None,
                    color: Option::None,
                    line_style: Option::None,
                    line_width: Option::None,
                    marker_style: Option::None,
                    marker_fill: Option::None,
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
                self.name = Option::Some(title.to_string());
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

#[derive(PartialEq, Clone)]
pub enum PlotAxisScaling {
    Linear,
    Log,
    NoAxis
}

// pub struct PlotLine2d<T> {
//     data_series: Vec<LineDataSeries2<T>>,
//     plot_width: u32,
//     plot_height: u32,
//     x_axis_scaling: Option<PlotAxisScaling>,
//     y_axis_scaling: Option<PlotAxisScaling>,
//     y2_axis_scaling: Option<PlotAxisScaling>,
//     title: Option<String>,
//     x_label: Option<String>,
//     y_label: Option<String>,
//     y2_label: Option<String>,
//     x_range: Option<Range<T>>,
//     y_range: Option<Range<T>>,
//     y2_range: Option<Range<T>>,
// }

// pub trait LineSeriesPlotFunctions2<T> {
//     fn new() -> Self;
//     fn add_simple_data_series(&mut self, x_data: Vec<T>, y_data: Vec<T>) -> &mut Self;
//     fn add_simple_data_series_y(&mut self, y_data: Vec<T>) -> &mut Self;
//     fn set_x_axis_type(&mut self, axis_type: PlotAxisScaling) -> &mut Self;
//     fn set_y_axis_type(&mut self, axis_type: PlotAxisScaling) -> &mut Self;
//     fn set_y2_axis_type(&mut self, axis_type: PlotAxisScaling) -> &mut Self;
//     fn set_title(&mut self, title: &str) -> &mut Self;
//     fn set_x_label(&mut self, title: &str) -> &mut Self;
//     fn set_y_label(&mut self, title: &str) -> &mut Self;
//     fn set_y2_label(&mut self, title: &str) -> &mut Self;
//     fn set_x_range(&mut self, range: Range<T>) -> &mut Self;
//     fn set_y_range(&mut self, range: Range<T>) -> &mut Self;
//     fn set_y2_range(&mut self, range: Range<T>) -> &mut Self;
// }

// macro_rules! impl_line_series_plot_functions_per_type {
//     ($T:ident) => {
//         impl LineSeriesPlotFunctions2<$T> for LineSeriesPlot2<$T> {
//             fn new() -> Self {
//                 Self {
//                     data_series: Vec::<LineDataSeries2<$T>>::new(),
//                     plot_width: STD_PLOT_WIDTH,
//                     plot_height: STD_PLOT_HEIGHT,
//                     x_axis_scaling: Option::None,
//                     y_axis_scaling: Option::None,
//                     y2_axis_scaling: Option::None,
//                     title: Option::None,
//                     x_label: Option::None,
//                     y_label: Option::None,
//                     y2_label: Option::None,
//                     x_range: Option::None,
//                     y_range: Option::None,
//                     y2_range: Option::None,
//                 }
//             }

//             fn add_simple_data_series(&mut self, x_data: Vec<$T>, y_data: Vec<$T>) -> &mut Self {
//                 self.data_series.push(LineDataSeries2::new(x_data, y_data));
//                 self
//             }

//             fn add_simple_data_series_y(&mut self, y_data: Vec<$T>) -> &mut Self {
//                 self.data_series.push(LineDataSeries2::new_y_only(y_data));
//                 self
//             }

//             fn set_x_axis_type(&mut self, axis_type: PlotAxisScaling) -> &mut Self {
//                 self.x_axis_scaling = Option::Some(axis_type);
//                 self
//             }

//             fn set_y_axis_type(&mut self, axis_type: PlotAxisScaling) -> &mut Self {
//                 self.y_axis_scaling = Option::Some(axis_type);
//                 self
//             }

//             fn set_y2_axis_type(&mut self, axis_type: PlotAxisScaling) -> &mut Self {
//                 self.y2_axis_scaling = Option::Some(axis_type);
//                 self
//             }

//             fn set_title(&mut self, title: &str) -> &mut Self {
//                 self.title = Option::Some(title.to_string());
//                 self
//             }

//             fn set_x_label(&mut self, title: &str) -> &mut Self {
//                 self.x_label = Option::Some(title.to_string());
//                 self
//             }

//             fn set_y_label(&mut self, title: &str) -> &mut Self {
//                 self.y_label = Option::Some(title.to_string());
//                 self
//             }

//             fn set_y2_label(&mut self, title: &str) -> &mut Self {
//                 self.y2_label = Option::Some(title.to_string());
//                 self
//             }

//             fn set_x_range(&mut self, range: Range<$T>) -> &mut Self {
//                 self.x_range = Option::Some(range);
//                 self
//             }

//             fn set_y_range(&mut self, range: Range<$T>) -> &mut Self {
//                 self.y_range = Option::Some(range);
//                 self
//             }

//             fn set_y2_range(&mut self, range: Range<$T>) -> &mut Self {
//                 self.y2_range = Option::Some(range);
//                 self
//             }
//         }
//     };
// }

pub struct PlotBuilder<T> {
    lines_2d: Option<Vec<Line2d<T>>>,
    plot_width: u32,
    plot_height: u32,
    x_axis_scaling: Option<PlotAxisScaling>,
    y_axis_scaling: Option<PlotAxisScaling>,
    y2_axis_scaling: Option<PlotAxisScaling>,
    z_axis_scaling: Option<PlotAxisScaling>,
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    y2_label: Option<String>,
    z_label: Option<String>,
    x_range: Option<Range<T>>,
    y_range: Option<Range<T>>,
    y2_range: Option<Range<T>>,
    z_range: Option<Range<T>>,
    plotting_settings: PlotStyleSettings,
}

pub enum SupportedPlotCombinations {
    Lines2d
}

pub trait PlotBuilderFunctions<T> {
    fn new() -> Self;
    fn new_sized() -> Self;
    fn set_x_axis_type(&mut self, axis_type: PlotAxisScaling) -> &mut Self;
    fn set_y_axis_type(&mut self, axis_type: PlotAxisScaling) -> &mut Self;
    fn set_y2_axis_type(&mut self, axis_type: PlotAxisScaling) -> &mut Self;
    fn set_z_axis_type(&mut self, axis_type: PlotAxisScaling) -> &mut Self;
    fn set_title(&mut self, title: &str) -> &mut Self;
    fn set_x_label(&mut self, title: &str) -> &mut Self;
    fn set_y_label(&mut self, title: &str) -> &mut Self;
    fn set_y2_label(&mut self, title: &str) -> &mut Self;
    fn set_z_label(&mut self, title: &str) -> &mut Self;
    fn set_x_range(&mut self, range: Range<T>) -> &mut Self;
    fn set_y_range(&mut self, range: Range<T>) -> &mut Self;
    fn set_y2_range(&mut self, range: Range<T>) -> &mut Self;
    fn set_z_range(&mut self, range: Range<T>) -> &mut Self;
    fn set_plot_width(&mut self, width: u32) -> &mut Self;
    fn set_plot_height(&mut self, height: u32) -> &mut Self;
    fn set_plot_size(&mut self, width: u32, height: u32) -> &mut Self;
    fn guarantee_2d_lines_initialized(&mut self) -> &mut Self;
    fn add_2d_line(&mut self, line_2d: &Line2d<T>) -> &mut Self;
    fn add_simple_2d_line(&mut self, x_data: &Vec<T>, y_data: &Vec<T>) -> &mut Self;
    fn add_simple_2d_line_y_only(&mut self, y_data: &Vec<T>) -> &mut Self;
    fn to_plot_processor_unitialized(self) -> PlotProcessor<T, NoPlotBackend>;
    fn to_plot_processor_unitialized_with_settings(self, settings: PlotStyleSettings) -> PlotProcessor<T, NoPlotBackend>;
    fn to_plotters_processor(self) -> PlotProcessor<T, PlottersBackend>;
    fn to_plotters_processor_with_settings(self, settings: PlotStyleSettings) -> PlotProcessor<T, PlottersBackend>;
    fn to_plotpy_processor(self) -> PlotProcessor<T, PlotPyBackend>;
    fn to_plotpy_processor_with_settings(self, settings: PlotStyleSettings) -> PlotProcessor<T, PlotPyBackend>;
}

macro_rules! impl_combined_plots_functions_per_type {
    ($T:ident) => {
        impl PlotBuilderFunctions<$T> for PlotBuilder<$T> {
            fn new() -> Self {
                Self {
                    lines_2d: Option::None,
                    plot_width: STD_PLOT_WIDTH,
                    plot_height: STD_PLOT_HEIGHT,
                    x_axis_scaling: Option::None,
                    y_axis_scaling: Option::None,
                    y2_axis_scaling: Option::None,
                    z_axis_scaling: Option::None,
                    title: Option::None,
                    x_label: Option::None,
                    y_label: Option::None,
                    y2_label: Option::None,
                    z_label: Option::None,
                    x_range: Option::None,
                    y_range: Option::None,
                    y2_range: Option::None,
                    z_range: Option::None,
                    plotting_settings: PlotStyleSettings::new(),
                }
            }

            fn new_sized() -> Self {
                let std_settings = Self::new();
                std_settings
            }

            fn set_x_axis_type(&mut self, axis_type: PlotAxisScaling) -> &mut Self {
                self.x_axis_scaling = Option::Some(axis_type);
                self
            }

            fn set_y_axis_type(&mut self, axis_type: PlotAxisScaling) -> &mut Self {
                self.y_axis_scaling = Option::Some(axis_type);
                self
            }

            fn set_y2_axis_type(&mut self, axis_type: PlotAxisScaling) -> &mut Self {
                self.y2_axis_scaling = Option::Some(axis_type);
                self
            }

            fn set_z_axis_type(&mut self, axis_type: PlotAxisScaling) -> &mut Self {
                self.z_axis_scaling = Option::Some(axis_type);
                self
            }

            fn set_title(&mut self, title: &str) -> &mut Self {
                self.title = Option::Some(title.to_string());
                self
            }

            fn set_x_label(&mut self, title: &str) -> &mut Self {
                self.x_label = Option::Some(title.to_string());
                self
            }

            fn set_y_label(&mut self, title: &str) -> &mut Self {
                self.y_label = Option::Some(title.to_string());
                self
            }

            fn set_y2_label(&mut self, title: &str) -> &mut Self {
                self.y2_label = Option::Some(title.to_string());
                self
            }

            fn set_z_label(&mut self, title: &str) -> &mut Self {
                self.z_label = Option::Some(title.to_string());
                self
            }

            fn set_x_range(&mut self, range: Range<$T>) -> &mut Self {
                self.x_range = Option::Some(range);
                self
            }

            fn set_y_range(&mut self, range: Range<$T>) -> &mut Self {
                self.y_range = Option::Some(range);
                self
            }

            fn set_y2_range(&mut self, range: Range<$T>) -> &mut Self {
                self.y2_range = Option::Some(range);
                self
            }

            fn set_z_range(&mut self, range: Range<$T>) -> &mut Self {
                self.z_range = Option::Some(range);
                self
            }

            fn set_plot_width(&mut self, width: u32) -> &mut Self {
                self.plot_width = width;
                self
            }

            fn set_plot_height(&mut self, height: u32) -> &mut Self {
                self.plot_height = height;
                self
            }

            fn set_plot_size(&mut self, width: u32, height: u32) -> &mut Self {
                self.set_plot_width(width).set_plot_height(height);
                self
            }

            fn guarantee_2d_lines_initialized(&mut self) -> &mut Self {
                // only call this function when a 2d Line will be added
                if let Option::None = self.lines_2d {
                    self.lines_2d = Option::Some(Vec::<Line2d<$T>>::new());
                }

                self
            }

            fn add_2d_line(&mut self, mut line_2d: &Line2d<$T>) -> &mut Self {
                self.guarantee_2d_lines_initialized();
                self.lines_2d.as_mut().unwrap().push(line_2d.clone());

                self
            }

            fn add_simple_2d_line(&mut self, x_data: &Vec<$T>, y_data: &Vec<$T>) -> &mut Self {
                self.guarantee_2d_lines_initialized();
                self.lines_2d.as_mut().unwrap().push(Line2d::new(x_data, y_data));
                self
            }

            fn add_simple_2d_line_y_only(&mut self, y_data: &Vec<$T>) -> &mut Self {
                self.guarantee_2d_lines_initialized();
                self.lines_2d.as_mut().unwrap().push(Line2d::new_y_only(y_data));
                self
            }

            fn to_plot_processor_unitialized(mut self) -> PlotProcessor<$T, NoPlotBackend> {
                self.deduce_ranges_and_axis_types();
                PlotProcessor::new_unitialized(self, PlotStyleSettings::new())
            }

            fn to_plot_processor_unitialized_with_settings(mut self, settings: PlotStyleSettings) -> PlotProcessor<$T, NoPlotBackend> {
                self.deduce_ranges_and_axis_types();
                PlotProcessor::new_unitialized(self, settings)
            }

            fn to_plotters_processor(mut self) -> PlotProcessor<$T, PlottersBackend> {
                self.deduce_ranges_and_axis_types();
                PlotProcessor::new_plotters_backend(self, PlotStyleSettings::new())
            }

            fn to_plotters_processor_with_settings(mut self, settings: PlotStyleSettings) -> PlotProcessor<$T, PlottersBackend> {
                self.deduce_ranges_and_axis_types();
                PlotProcessor::new_plotters_backend(self, settings)
            }

            fn to_plotpy_processor(mut self) -> PlotProcessor<$T, PlotPyBackend> {
                self.deduce_ranges_and_axis_types();
                PlotProcessor::new_plotpy_backend(self, PlotStyleSettings::new())
            }

            fn to_plotpy_processor_with_settings(mut self, settings: PlotStyleSettings) -> PlotProcessor<$T, PlotPyBackend> {
                self.deduce_ranges_and_axis_types();
                PlotProcessor::new_plotpy_backend(self, settings)
            }
    }
    };
}

trait PlotsPreProcessFunctions<T> {
    fn deduce_ranges_and_axis_types(&mut self) -> &mut Self;
    fn indentify_plot_type(&self) -> SupportedPlotCombinations;
}

macro_rules! impl_combined_plots_known_range_functions_per_type {
    ($T:ident) => {
        impl PlotsPreProcessFunctions<$T> for PlotBuilder<$T> {
            fn deduce_ranges_and_axis_types(&mut self) -> &mut Self {
                let mut x_min = $T::MAX;
                let mut x_max = $T::MIN;
                let mut y_min = $T::MAX;
                let mut y_max = $T::MIN;
                let mut y2_min = $T::MAX;
                let mut y2_max = $T::MIN;
                let mut z_min = $T::MAX;
                let mut z_max = $T::MIN;

                let mut found_any_on_y2 = false;
                let mut found_any_on_z = false;

                if let Option::Some(lines_2d) = &self.lines_2d {
                    for line_2d in lines_2d {

                        for x_value in &line_2d.x_values {
                            if *x_value < x_min {
                                x_min = *x_value
                            } else if *x_value > x_max {
                                x_max = *x_value
                            }
                        }

                        if line_2d.on_secondary_axis { 
                            found_any_on_y2 = true;
                            for y_value in &line_2d.y_values {
                                if *y_value < y2_min {
                                    y2_min = *y_value
                                } else if *y_value > y2_max {
                                    y2_max = *y_value
                                }
                            }

                        } else {
                            for y_value in &line_2d.y_values {
                                if *y_value < y_min {
                                    y_min = *y_value
                                } else if *y_value > y_max {
                                    y_max = *y_value
                                }
                            }
                        }
                    }
                } else { // include deduce ranges for all new types of plots that can be added.
                    // if things can be found on the z axis also note that
                }

                // store found values only if user did not set limits yet!
                if let Option::None = self.x_range {
                    self.x_range = Option::Some(x_min..x_max);
                }
                if let Option::None = self.y_range {
                    self.y_range = Option::Some(y_min..y_max);
                }
                if let Option::None = self.y2_range {
                    self.y2_range = Option::Some(y2_min..y2_max);
                }
                if let Option::None = self.z_range {
                    self.z_range = Option::Some(z_min..z_max);
                }

                // set axis type to linear on all axis that have values if not set yet
                // x and y axis are allways used (i presume)
                if let Option::None = self.x_axis_scaling {
                    self.x_axis_scaling = Option::Some(PlotAxisScaling::Linear);
                }
                if let Option::None = self.y_axis_scaling {
                    self.y_axis_scaling = Option::Some(PlotAxisScaling::Linear);
                }
                if let Option::None = self.y2_axis_scaling {
                    if found_any_on_y2 {
                        self.y2_axis_scaling = Option::Some(PlotAxisScaling::Linear);
                    } else {
                        self.y2_axis_scaling = Option::Some(PlotAxisScaling::NoAxis);
                    }
                }
                if let Option::None = self.z_axis_scaling {
                    if found_any_on_z {
                        self.z_axis_scaling = Option::Some(PlotAxisScaling::Linear);
                    } else {
                        self.z_axis_scaling = Option::Some(PlotAxisScaling::NoAxis);
                    }
                }

                // in the end all axis types and ranges should be of type Option::Some(_)

                self
            }

            fn indentify_plot_type(&self) -> SupportedPlotCombinations {

                // get bool of each
                let plot_2d_included: bool;
                if let Some(_) = self.lines_2d {
                    plot_2d_included = true;
                } else {
                    plot_2d_included = false;
                }

                // match the supported combinations
                match (plot_2d_included) {
                    (true) => {                       
                        SupportedPlotCombinations::Lines2d
                    },
                    (false) => { // AKA all other options, meant to be _ => { but left as example
                        println!("Found combination of plots: plot_2d = {}", plot_2d_included);
                        panic!("This combination of plots is not supported!");
                    }, 
                }
            }
        }
    };
}

pub struct NoPlotBackend {}
pub struct PlotPyBackend {}
pub struct  PlottersBackend {}

// pub trait 

pub struct PlotProcessor<T, Backend> {
    plots: PlotBuilder<T>,
    backend: PhantomData<Backend>,
    plot_style_settings: PlotStyleSettings
}

impl<T> PlotProcessor<T, NoPlotBackend> {
    pub fn new_unitialized(plots: PlotBuilder<T>, plot_style_settings: PlotStyleSettings) -> Self {
        Self {plots, backend:PhantomData::<NoPlotBackend>, plot_style_settings}
    }

    pub fn use_plotters_backend(self) -> PlotProcessor<T, PlottersBackend> {
        PlotProcessor { plots: self.plots, backend: PhantomData::<PlottersBackend>, plot_style_settings: self.plot_style_settings}
    }

    pub fn use_plotpy_backend(self) -> PlotProcessor<T, PlotPyBackend> {
        PlotProcessor { plots: self.plots, backend: PhantomData::<PlotPyBackend>, plot_style_settings: self.plot_style_settings }
    }
}
impl<T> PlotProcessor<T, PlottersBackend> {
    pub fn new_plotters_backend(plots: PlotBuilder<T>, plot_style_settings: PlotStyleSettings) -> Self {
        Self {plots, backend:PhantomData::<PlottersBackend>, plot_style_settings}
    }
}

impl<T> PlotProcessor<T, PlotPyBackend> {
    pub fn new_plotpy_backend(plots: PlotBuilder<T>, plot_style_settings: PlotStyleSettings) -> Self {
        Self {plots, backend:PhantomData::<PlotPyBackend>, plot_style_settings}
    }
}



pub trait PlotProcessorPlottersBackendFunctions<T> {
    fn process(&self, root: &mut DrawingArea<SVGBackend, Shift>);
    fn SVG_to_file(&self, file_path: &str);
    fn SVG_to_mem(&self) -> String;
    fn SVG_to_RGBA(&self) -> Box<[u8]>;

    fn simple_2d_plot(&self, root: &mut DrawingArea<SVGBackend, Shift>);
}


macro_rules! impl_plot_processor_plotters_backend_functions_per_type {
    ($T: ident) => {
        impl PlotProcessorPlottersBackendFunctions<$T> for PlotProcessor<$T, PlottersBackend> {
            fn process(&self, root: &mut DrawingArea<SVGBackend, Shift>) {

                // Some standard settings
                root.margin(self.plot_style_settings.plotters_figure_padding, self.plot_style_settings.plotters_figure_padding, self.plot_style_settings.plotters_figure_padding,self.plot_style_settings.plotters_figure_padding)
                    .fill(&WHITE).unwrap();

                match (self.plots.indentify_plot_type() ){
                    (SupportedPlotCombinations::Lines2d) => {
                        self.simple_2d_plot(root)
                    },
                    _ => {panic!("This plot type is not supported by the PLOTTERS backend")}
                }

                // indentify the kind of plot

                // do some standard things

                // add feature for all the things that are in there

                // call .present() to make it all happen
                root.present().unwrap();

                // done
            }

            fn SVG_to_file(&self, file_path: &str) {
                let mut root = SVGBackend::new(file_path, (self.plots.plot_width, self.plots.plot_height)).into_drawing_area();
                self.process(&mut root);
            }

            fn SVG_to_mem(&self) -> String {
                let mut string_buffer = String::new();
                {
                    let mut root = SVGBackend::with_string(&mut string_buffer, (self.plots.plot_width, self.plots.plot_height)).into_drawing_area();
                    self.process(&mut root);
                }
                string_buffer
            }
            
            fn SVG_to_RGBA(&self) -> Box<[u8]> {
                // probably this one utilizes SVG_to_mem first
                todo!()
            }

            fn simple_2d_plot(&self, root: &mut DrawingArea<SVGBackend, Shift>) {

                let title_of_figure: String;
                match &self.plots.title {
                    // Option::Some(title) => {chart.caption(title, ("sans-serif", 50.0).into_font());}
                    Option::Some(title) => {title_of_figure = title.to_string()}
                    _ => {title_of_figure = self.plot_style_settings.title.clone()}
                }

                let range_x_f64 = self.plots.x_range.as_ref().unwrap().start as f64 .. self.plots.x_range.as_ref().unwrap().end as f64;
                let range_y_f64 = self.plots.y_range.as_ref().unwrap().start as f64 .. self.plots.y_range.as_ref().unwrap().end as f64;
                let range_y2_f64 = self.plots.y2_range.as_ref().unwrap().start as f64 .. self.plots.y2_range.as_ref().unwrap().end as f64;
                // let range_y2_f64_dummy = 1. as f64 .. 1.1 as f64;

                // using unwrap 
                let x_axis_scaling = self.plots.x_axis_scaling.clone().expect("x axis scaling should have been set!");
                let y_axis_scaling = self.plots.y_axis_scaling.clone().expect("y axis scaling should have been set!");
                let y2_axis_scaling = self.plots.y2_axis_scaling.clone().expect("y2 axis scaling should have been set!");

                // get ready for some an-idiometicness
                match (x_axis_scaling, y_axis_scaling, y2_axis_scaling) {
                    (PlotAxisScaling::Linear, PlotAxisScaling::Linear, PlotAxisScaling::NoAxis) => {
                        let mut chart = ChartBuilder::on(&root)
                            .x_label_area_size(self.plot_style_settings.plotters_x_label_area_size)
                            .y_label_area_size(self.plot_style_settings.plotters_y_label_area_size)
                            .right_y_label_area_size(self.plot_style_settings.plotters_right_y_label_area_size)
                            .margin(self.plot_style_settings.plotters_margin)
                            .caption(title_of_figure, ("sans-serif", 50.0).into_font())
                            .build_cartesian_2d(range_x_f64.clone(), range_y_f64).unwrap()
                            .set_secondary_coord(range_x_f64, range_y2_f64);
                        
                        impl_rest_of_plot_settings!(chart, self);
                    },
                    (PlotAxisScaling::Log, PlotAxisScaling::Linear, PlotAxisScaling::NoAxis) => {
                        let mut chart = ChartBuilder::on(&root)
                            .x_label_area_size(self.plot_style_settings.plotters_x_label_area_size)
                            .y_label_area_size(self.plot_style_settings.plotters_y_label_area_size)
                            .right_y_label_area_size(self.plot_style_settings.plotters_right_y_label_area_size)
                            .margin(self.plot_style_settings.plotters_margin)
                            .caption(title_of_figure, ("sans-serif", 50.0).into_font())
                            .build_cartesian_2d((range_x_f64.clone()).log_scale(), range_y_f64).unwrap()
                            .set_secondary_coord(range_x_f64, range_y2_f64);
                        
                        impl_rest_of_plot_settings!(chart, self);
                    },
                    (PlotAxisScaling::Linear, PlotAxisScaling::Log, PlotAxisScaling::NoAxis) => {
                        let mut chart = ChartBuilder::on(&root)
                            .x_label_area_size(self.plot_style_settings.plotters_x_label_area_size)
                            .y_label_area_size(self.plot_style_settings.plotters_y_label_area_size)
                            .right_y_label_area_size(self.plot_style_settings.plotters_right_y_label_area_size)
                            .margin(self.plot_style_settings.plotters_margin)
                            .caption(title_of_figure, ("sans-serif", 50.0).into_font())
                            .build_cartesian_2d(range_x_f64.clone(), (range_y_f64).log_scale()).unwrap()
                            .set_secondary_coord(range_x_f64, range_y2_f64);
                        
                        impl_rest_of_plot_settings!(chart, self);
                    },
                    (PlotAxisScaling::Log, PlotAxisScaling::Log, PlotAxisScaling::NoAxis) => {
                        let mut chart = ChartBuilder::on(&root)
                            .x_label_area_size(self.plot_style_settings.plotters_x_label_area_size)
                            .y_label_area_size(self.plot_style_settings.plotters_y_label_area_size)
                            .right_y_label_area_size(self.plot_style_settings.plotters_right_y_label_area_size)
                            .margin(self.plot_style_settings.plotters_margin)
                            .caption(title_of_figure, ("sans-serif", 50.0).into_font())
                            .build_cartesian_2d((range_x_f64.clone()).log_scale(), (range_y_f64).log_scale()).unwrap()
                            .set_secondary_coord(range_x_f64, range_y2_f64);

                        impl_rest_of_plot_settings!(chart, self);
                    },
                    (PlotAxisScaling::Linear, PlotAxisScaling::Linear, PlotAxisScaling::Linear) => {
                        let mut chart = ChartBuilder::on(&root)
                            .x_label_area_size(self.plot_style_settings.plotters_x_label_area_size)
                            .y_label_area_size(self.plot_style_settings.plotters_y_label_area_size)
                            .right_y_label_area_size(self.plot_style_settings.plotters_right_y_label_area_size)
                            .margin(self.plot_style_settings.plotters_margin)
                            .caption(title_of_figure, ("sans-serif", 50.0).into_font())
                            .build_cartesian_2d(range_x_f64.clone(), range_y_f64).unwrap()
                            .set_secondary_coord(range_x_f64, range_y2_f64);

                        impl_rest_of_plot_settings!(chart, self);
                    },
                    (PlotAxisScaling::Linear, PlotAxisScaling::Linear, PlotAxisScaling::Log) => {
                        let mut chart = ChartBuilder::on(&root)
                            .x_label_area_size(self.plot_style_settings.plotters_x_label_area_size)
                            .y_label_area_size(self.plot_style_settings.plotters_y_label_area_size)
                            .right_y_label_area_size(self.plot_style_settings.plotters_right_y_label_area_size)
                            .margin(self.plot_style_settings.plotters_margin)
                            .caption(title_of_figure, ("sans-serif", 50.0).into_font())
                            .build_cartesian_2d(range_x_f64.clone(), range_y_f64).unwrap()
                            .set_secondary_coord(range_x_f64, (range_y2_f64).log_scale());
                        impl_rest_of_plot_settings!(chart, self);
                    },
                    (PlotAxisScaling::Log, PlotAxisScaling::Linear, PlotAxisScaling::Linear) => {
                        let mut chart = ChartBuilder::on(&root)
                            .x_label_area_size(self.plot_style_settings.plotters_x_label_area_size)
                            .y_label_area_size(self.plot_style_settings.plotters_y_label_area_size)
                            .right_y_label_area_size(self.plot_style_settings.plotters_right_y_label_area_size)
                            .margin(self.plot_style_settings.plotters_margin)
                            .caption(title_of_figure, ("sans-serif", 50.0).into_font())
                            .build_cartesian_2d((range_x_f64.clone()).log_scale(), range_y_f64).unwrap()
                            .set_secondary_coord((range_x_f64).log_scale(), range_y2_f64);

                        impl_rest_of_plot_settings!(chart, self);
                    },
                    (PlotAxisScaling::Log, PlotAxisScaling::Linear, PlotAxisScaling::Log) => {
                        let mut chart = ChartBuilder::on(&root)
                            .x_label_area_size(self.plot_style_settings.plotters_x_label_area_size)
                            .y_label_area_size(self.plot_style_settings.plotters_y_label_area_size)
                            .right_y_label_area_size(self.plot_style_settings.plotters_right_y_label_area_size)
                            .margin(self.plot_style_settings.plotters_margin)
                            .caption(title_of_figure, ("sans-serif", 50.0).into_font())
                            .build_cartesian_2d((range_x_f64.clone()).log_scale(), range_y_f64).unwrap()
                            .set_secondary_coord((range_x_f64).log_scale(), (range_y2_f64).log_scale());

                        impl_rest_of_plot_settings!(chart, self);
                    },
                    (PlotAxisScaling::Linear, PlotAxisScaling::Log, PlotAxisScaling::Linear) => {
                        let mut chart = ChartBuilder::on(&root)
                            .x_label_area_size(self.plot_style_settings.plotters_x_label_area_size)
                            .y_label_area_size(self.plot_style_settings.plotters_y_label_area_size)
                            .right_y_label_area_size(self.plot_style_settings.plotters_right_y_label_area_size)
                            .margin(self.plot_style_settings.plotters_margin)
                            .caption(title_of_figure, ("sans-serif", 50.0).into_font())
                            .build_cartesian_2d(range_x_f64.clone(), (range_y_f64).log_scale()).unwrap()
                            .set_secondary_coord(range_x_f64, range_y2_f64);

                        impl_rest_of_plot_settings!(chart, self);
                    },
                    (PlotAxisScaling::Linear, PlotAxisScaling::Log, PlotAxisScaling::Log) => {
                        let mut chart = ChartBuilder::on(&root)
                            .x_label_area_size(self.plot_style_settings.plotters_x_label_area_size)
                            .y_label_area_size(self.plot_style_settings.plotters_y_label_area_size)
                            .right_y_label_area_size(self.plot_style_settings.plotters_right_y_label_area_size)
                            .margin(self.plot_style_settings.plotters_margin)
                            .caption(title_of_figure, ("sans-serif", 50.0).into_font())
                            .build_cartesian_2d(range_x_f64.clone(), (range_y_f64).log_scale()).unwrap()
                            .set_secondary_coord(range_x_f64, (range_y2_f64).log_scale());

                        impl_rest_of_plot_settings!(chart, self);
                    },
                    (PlotAxisScaling::Log, PlotAxisScaling::Log, PlotAxisScaling::Linear) => {
                        let mut chart = ChartBuilder::on(&root)
                            .x_label_area_size(self.plot_style_settings.plotters_x_label_area_size)
                            .y_label_area_size(self.plot_style_settings.plotters_y_label_area_size)
                            .right_y_label_area_size(self.plot_style_settings.plotters_right_y_label_area_size)
                            .margin(self.plot_style_settings.plotters_margin)
                            .caption(title_of_figure, ("sans-serif", 50.0).into_font())
                            .build_cartesian_2d((range_x_f64.clone()).log_scale(), (range_y_f64).log_scale()).unwrap()
                            .set_secondary_coord((range_x_f64).log_scale(), range_y2_f64);

                        impl_rest_of_plot_settings!(chart, self);
                    },
                    (PlotAxisScaling::Log, PlotAxisScaling::Log, PlotAxisScaling::Log) => {
                        let mut chart = ChartBuilder::on(&root)
                            .x_label_area_size(self.plot_style_settings.plotters_x_label_area_size)
                            .y_label_area_size(self.plot_style_settings.plotters_y_label_area_size)
                            .right_y_label_area_size(self.plot_style_settings.plotters_right_y_label_area_size)
                            .margin(self.plot_style_settings.plotters_margin)
                            .caption(title_of_figure, ("sans-serif", 50.0).into_font())
                            .build_cartesian_2d((range_x_f64.clone()).log_scale(), (range_y_f64).log_scale()).unwrap()
                            .set_secondary_coord((range_x_f64).log_scale(), (range_y2_f64).log_scale());

                        impl_rest_of_plot_settings!(chart, self);
                    },
                    _ => {panic!("Either there is a genuine axis combination missing or something is messed up! :(")}
                }
            }
        }
    }
}


macro_rules! impl_rest_of_plot_settings {
    ($chart: expr, $self: expr) => {
        $chart.configure_mesh().draw().unwrap(); // do some more work on this

        match $self.plots.y2_axis_scaling.as_ref().unwrap() {
            PlotAxisScaling::NoAxis => {},
            _ => {
                // process the settings for the secondary axis, maybe this can be made nicer
                let secondary_axis_label: String;
                if let Option::Some(label) = $self.plots.y2_label.clone() {
                    secondary_axis_label = label;
                } else {
                    secondary_axis_label = String::new();
                }

                $chart
                    .configure_secondary_axes()
                    .y_desc(secondary_axis_label)
                    .draw().unwrap();
            }
        }

        let amount_of_lines = $self.plots.lines_2d.as_ref().unwrap().len();
        for (number, series) in $self.plots.lines_2d.as_ref().unwrap().iter().enumerate() {
            let legend_name: String;
            if let Option::Some(series_name) = &series.name {
                legend_name = series_name.clone();
            } else {
                legend_name = { format!{"Data_series {number}"} }
            }


            let line_width: u32;
            if let Option::Some(prefered_line_width) = series.line_width {
                line_width = prefered_line_width;
            } else {
                line_width = $self.plot_style_settings.line_width;
            }

            let marker_size: u32;
            if let Option::Some(prefered_style) = &series.marker_style {
                match prefered_style {
                    MarkerStyle::None => {marker_size = 0},
                    MarkerStyle::Circle(size) => {marker_size = *size},
                    MarkerStyle::Diamand(size) => {marker_size = *size},
                    MarkerStyle::Cross(size) => {marker_size = *size},
                }
            } else {
                match $self.plot_style_settings.marker_style {
                    MarkerStyle::None => {marker_size = 0},
                    MarkerStyle::Circle(size) => {marker_size = size},
                    MarkerStyle::Diamand(size) => {marker_size = size},
                    MarkerStyle::Cross(size) => {marker_size = size},
                }
            }

            let marker_fill: bool;
            if let Option::Some(marker_fill_option) = &series.marker_fill {
                match marker_fill_option {
                    MarkerFill::Filled => {
                        marker_fill = true;
                    },
                    MarkerFill::NotFilled => {
                        marker_fill = false;
                    }
                }
            } else {
                match $self.plot_style_settings.marker_fill {
                    MarkerFill::Filled => {
                        marker_fill = true;
                    },
                    MarkerFill::NotFilled => {
                        marker_fill = false;
                    }
                }
            }

            let line_color: RGBAColor;
            if let Option::Some(color) = &series.color {
                line_color = *color;
            } else {
                if let Option::Some(color) = $self.plot_style_settings.line_color {
                    line_color = color;
                } else {
                    line_color = $self.plot_style_settings.color_map.get_color(number as f32/ amount_of_lines as f32);
                }
            }

            let style = ShapeStyle {color: line_color, filled: marker_fill, stroke_width: line_width};

            if !(series.on_secondary_axis) {
                $chart
                    .draw_series(LineSeries::new(series.x_values.iter().zip(series.y_values.clone()).map(|(x, y)| (*x as f64, y as f64)), style)
                    .point_size(marker_size))
                    .unwrap()
                    .label(legend_name);
            } else {
                $chart
                    .draw_secondary_series(LineSeries::new(series.x_values.iter().zip(series.y_values.clone()).map(|(x, y)| (*x as f64, y as f64)), style)
                    .point_size(marker_size))
                    .unwrap()
                    .label(legend_name);
            }            
        }
    }
}


// impl_line_series_plot_functions_per_type!(i8);
// impl_line_series_plot_functions_per_type!(i16);
// impl_line_series_plot_functions_per_type!(i32);
// impl_line_series_plot_functions_per_type!(i64);

// impl_line_series_plot_functions_per_type!(isize);

// impl_line_series_plot_functions_per_type!(f32);
// impl_line_series_plot_functions_per_type!(f64);


impl_line_data_series_functions_per_type!(i8);
impl_line_data_series_functions_per_type!(i16);
impl_line_data_series_functions_per_type!(i32);
impl_line_data_series_functions_per_type!(i64);

impl_line_data_series_functions_per_type!(isize);

impl_line_data_series_functions_per_type!(f32);
impl_line_data_series_functions_per_type!(f64);

impl_combined_plots_functions_per_type!(i8);
impl_combined_plots_functions_per_type!(i16);
impl_combined_plots_functions_per_type!(i32);
impl_combined_plots_functions_per_type!(i64);

impl_combined_plots_functions_per_type!(isize);

impl_combined_plots_functions_per_type!(f32);
impl_combined_plots_functions_per_type!(f64);

impl_plot_processor_plotters_backend_functions_per_type!(i8);
impl_plot_processor_plotters_backend_functions_per_type!(i16);
impl_plot_processor_plotters_backend_functions_per_type!(i32);
impl_plot_processor_plotters_backend_functions_per_type!(i64);

impl_plot_processor_plotters_backend_functions_per_type!(isize);

impl_plot_processor_plotters_backend_functions_per_type!(f32);
impl_plot_processor_plotters_backend_functions_per_type!(f64);

impl_combined_plots_known_range_functions_per_type!(i8);
impl_combined_plots_known_range_functions_per_type!(i16);
impl_combined_plots_known_range_functions_per_type!(i32);
impl_combined_plots_known_range_functions_per_type!(i64);

impl_combined_plots_known_range_functions_per_type!(isize);

impl_combined_plots_known_range_functions_per_type!(f32);
impl_combined_plots_known_range_functions_per_type!(f64);
// pub struct PlottersBackendInitiator<V> {
//     storage: V
// }

// pub trait PlottersBackendInitiatorFunctions<> for 

// pub struct PlottersBackend<T, V> {
//     backend: T,
//     storage: V
// }

// pub trait PlottersBackendSGVFunctions<V> {
//     fn new_svg_backend(path: V) -> Self;
//     fn new_svg_backend_sized(path: V, width: u32, height: u32) -> Self;
// }

// impl PlottersBackendSGVFunctions<&str> for PlottersBackend<SVGBackend<'_>, &str> {
//     fn new_svg_backend(path: &str) -> Self {
//         PlottersBackend { backend: SVGBackend::new(path, (500, 500)), storage: path}
//     }

//     fn new_svg_backend_sized(path: &str, width: u32, height: u32) -> Self {
//         PlottersBackend { backend: SVGBackend::new(path, (width, height)), storage: path}
//     }
// }

// impl PlottersBackendSGVFunctions<&str> for PlottersBackend<SVGBackend<'_>, String> {
//     fn new_svg_backend(string_in_memory: String) -> Self {
//         PlottersBackend { backend: SVGBackend::new(&mut string_in_memory, (500, 500)), storage: string_in_memory}
//     }

//     fn new_svg_backend_sized(path: &str, width: u32, height: u32) -> Self {
//         PlottersBackend { backend: SVGBackend::new(path, (width, height)), storage: path}
//     }
// }

// fn some_fun() {
//     let x = SVGBackend::new
// }