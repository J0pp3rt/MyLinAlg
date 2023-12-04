use crate::plotting::{*};


#[derive(Clone)]
pub struct PlotBuilder<T> {
    pub lines_2d: Option<Vec<Line2d<T>>>,
    pub lines_3d: Option<Vec<Line3d<T>>>,
    pub surface_3d: Option<Surface3d<T>>,
    pub plot_type: Option<SupportedPlotTypes>,
    pub plot_width: Option<u32>,
    pub plot_height: Option<u32>,
    pub x_axis_scaling: Option<PlotAxisScaling>,
    pub y_axis_scaling: Option<PlotAxisScaling>,
    pub y2_axis_scaling: Option<PlotAxisScaling>,
    pub z_axis_scaling: Option<PlotAxisScaling>,
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub y2_label: String,
    pub z_label: String,
    pub x_range: Option<Range<T>>,
    pub y_range: Option<Range<T>>,
    pub y2_range: Option<Range<T>>,
    pub z_range: Option<Range<T>>,
    pub plotting_settings: PlotStyleSettings,
}

#[derive(Clone)]
pub enum SupportedPlotTypes {
    Lines2d,
    Lines3d,
    HeatMap,
    HeatMapAndLines2d,
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
    fn set_plot_type(&mut self, plot_type: SupportedPlotTypes) -> &mut Self;
    fn guarantee_2d_lines_initialized(&mut self) -> &mut Self;
    fn add_2d_line(&mut self, line_2d: &Line2d<T>) -> &mut Self;
    fn add_simple_2d_line(&mut self, x_data: &Vec<T>, y_data: &Vec<T>) -> &mut Self;
    fn add_simple_2d_line_y_only(&mut self, y_data: &Vec<T>) -> &mut Self;
    fn guarantee_3d_lines_initialized(&mut self) -> &mut Self;
    fn add_3d_line(&mut self, line_3d: &Line3d<T>) -> &mut Self;
    fn add_simple_3d_line(&mut self, x_data: &Vec<T>, y_data: &Vec<T>, z_data: &Vec<T>) -> &mut Self;
    fn add_surface_3d(&mut self, surface_3d: Surface3d<T>) -> &mut Self;
    fn add_surface_plot_xyz(&mut self, x_data: &Vec<T>, y_data: &Vec<T>, z_data: &Vec<Vec<T>>) -> &mut Self;
    fn to_plot_processor_unitialized(self) -> PlotProcessor<T, NoPlotBackend>;
    fn to_plotters_processor(self) -> PlotProcessor<T, PlottersBackend>;
    fn to_plotpy_processor(self) -> PlotProcessor<T, PlotPyBackend>;
    fn get_plot_dimensions(&self) -> (u32, u32);
}

macro_rules! impl_combined_plots_functions_per_type {
    ($T:ident) => {
        impl PlotBuilderFunctions<$T> for PlotBuilder<$T> {
            fn new() -> Self {
                Self {
                    lines_2d: Option::None,
                    lines_3d: Option::None,
                    surface_3d: Option::None,
                    plot_type: Option::None,
                    plot_width: Option::None,
                    plot_height: Option::None,
                    x_axis_scaling: Option::None,
                    y_axis_scaling: Option::None,
                    y2_axis_scaling: Option::None,
                    z_axis_scaling: Option::None,
                    title: String::new(),
                    x_label: String::new(),
                    y_label: String::new(),
                    y2_label: String::new(),
                    z_label: String::new(),
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
                self.title = title.to_string();
                self
            }

            fn set_x_label(&mut self, title: &str) -> &mut Self {
                self.x_label = title.to_string();
                self
            }

            fn set_y_label(&mut self, title: &str) -> &mut Self {
                self.y_label = title.to_string();
                self
            }

            fn set_y2_label(&mut self, title: &str) -> &mut Self {
                self.y2_label = title.to_string();
                self
            }

            fn set_z_label(&mut self, title: &str) -> &mut Self {
                self.z_label = title.to_string();
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
                self.plot_width = Option::Some(width);
                self
            }

            fn set_plot_height(&mut self, height: u32) -> &mut Self {
                self.plot_height = Option::Some(height);
                self
            }

            fn set_plot_size(&mut self, width: u32, height: u32) -> &mut Self {
                self.set_plot_width(width).set_plot_height(height);
                self
            }

            fn set_plot_type(&mut self, plot_type: SupportedPlotTypes) -> &mut Self {
                self.plot_type = Option::Some(plot_type);
                self
            }

            fn guarantee_2d_lines_initialized(&mut self) -> &mut Self {
                // only call this function when a 2d Line will be added
                if let Option::None = self.lines_2d {
                    self.lines_2d = Option::Some(Vec::<Line2d<$T>>::new());
                }

                self
            }

            fn add_2d_line(&mut self, line_2d: &Line2d<$T>) -> &mut Self {
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

            fn guarantee_3d_lines_initialized(&mut self) -> &mut Self {
                // only call this function when a 2d Line will be added
                if let Option::None = self.lines_3d {
                    self.lines_3d = Option::Some(Vec::<Line3d<$T>>::new());
                }

                self
            }

            fn add_3d_line(&mut self, line_3d: &Line3d<$T>) -> &mut Self {
                self.guarantee_3d_lines_initialized();
                self.lines_3d.as_mut().unwrap().push(line_3d.clone());

                self
            }

            fn add_simple_3d_line(&mut self, x_data: &Vec<$T>, y_data: &Vec<$T>, z_data: &Vec<$T>) -> &mut Self {
                self.guarantee_3d_lines_initialized();
                self.lines_3d.as_mut().unwrap().push(Line3d::new(x_data, y_data, z_data));
                self
            }

            fn add_surface_3d(&mut self, surface_3d: Surface3d<$T>) -> &mut Self {
                self.surface_3d = Option::Some(surface_3d);

                self
            }

            fn add_surface_plot_xyz(&mut self, x_data: &Vec<$T>, y_data: &Vec<$T>, z_data: &Vec<Vec<$T>>) -> &mut Self {
                let surface_plot = Surface3d::new(x_data, y_data, z_data);
                self.add_surface_3d(surface_plot);
                self
            }

            fn to_plot_processor_unitialized(mut self) -> PlotProcessor<$T, NoPlotBackend> {
                self.deduce_ranges_and_axis_types();
                PlotProcessor::new_unitialized(self)
            }

            fn to_plotters_processor(mut self) -> PlotProcessor<$T, PlottersBackend> {
                self.deduce_ranges_and_axis_types();
                PlotProcessor::new_plotters_backend(self)
            }

            fn to_plotpy_processor(mut self) -> PlotProcessor<$T, PlotPyBackend> {
                self.deduce_ranges_and_axis_types();
                PlotProcessor::new_plotpy_backend(self)
            }

            fn get_plot_dimensions(&self) -> (u32, u32) {
                // returns (width, height)
                let width: u32;
                let height: u32;
                match self.plot_width {
                    Option::Some(prefered_width) => {
                        width = prefered_width;
                    }, 
                    Option::None => {
                        width = self.plotting_settings.plot_width;
                    }
                }

                match self.plot_height {
                    Option::Some(prefered_height) => {
                        height = prefered_height;
                    }, 
                    Option::None => {
                        height = self.plotting_settings.plot_height;
                    }
                }

                (width, height)
            }
    }
    };
}


pub trait PlotsPreProcessFunctions<T> {
    fn deduce_ranges_and_axis_types(&mut self) -> &mut Self;
    fn indentify_plot_type(&self) -> SupportedPlotTypes;
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
                            }
                            if *x_value > x_max {
                                x_max = *x_value
                            }
                        }

                        if line_2d.on_secondary_axis { 
                            found_any_on_y2 = true;
                            for y_value in &line_2d.y_values {
                                if *y_value < y2_min {
                                    y2_min = *y_value
                                }
                                if *y_value > y2_max {
                                    y2_max = *y_value
                                }
                            }

                        } else {
                            for y_value in &line_2d.y_values {
                                if *y_value < y_min {
                                    y_min = *y_value
                                }
                                if *y_value > y_max {
                                    y_max = *y_value
                                }
                            }
                        }
                    }
                } 
                if let Option::Some(lines_3d) = &self.lines_3d { 
                    found_any_on_z = true;
                    for line_3d in lines_3d {
                        for x_value in &line_3d.x_values {
                            if *x_value < x_min {
                                x_min = *x_value
                            }
                            if *x_value > x_max {
                                x_max = *x_value
                            }
                        }

                        for y_value in &line_3d.y_values {
                            if *y_value < y_min {
                                y_min = *y_value
                            }
                            if *y_value > y_max {
                                y_max = *y_value
                            }
                        }

                        for z_value in &line_3d.z_values {
                            if *z_value < z_min {
                                z_min = *z_value
                            }
                            if *z_value > z_max {
                                z_max = *z_value
                            }
                        }

                    }
                } 
                if let Option::Some(surface_3d) = &self.surface_3d { 
                    found_any_on_z = true;
                        for x_value in &surface_3d.x_values {
                            if *x_value < x_min {
                                x_min = *x_value
                            }
                            if *x_value > x_max {
                                x_max = *x_value
                            }
                        }

                        for y_value in &surface_3d.y_values {
                            if *y_value < y_min {
                                y_min = *y_value
                            }
                            if *y_value > y_max {
                                y_max = *y_value
                            }
                        }

                        for z_value_row in &surface_3d.z_values {
                            for z_value in z_value_row {
                                if *z_value < z_min {
                                    z_min = *z_value
                                }
                                if *z_value > z_max {
                                    z_max = *z_value
                                }
                            }
                        }
                } 

                
                // else {// include deduce ranges for all new types of plots that can be added.
                //     // if things can be found on the z axis also note that

                // }

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

            fn indentify_plot_type(&self) -> SupportedPlotTypes {

                if let Option::Some(prefered_plot_type) = &self.plot_type {
                    match prefered_plot_type {
                        SupportedPlotTypes::Lines2d => {
                            if let Option::None = self.lines_2d {
                                panic!("No 2d lines initialized, can not make 2d line plot")
                            } else {
                                SupportedPlotTypes::Lines2d
                            }
                        },
                        SupportedPlotTypes::Lines3d => {
                            if let Option::None = self.lines_3d {
                                panic!("No 3d lines initialized, can not make 3d line plot")
                            } else {
                                SupportedPlotTypes::Lines3d
                            }
                        },
                        SupportedPlotTypes::HeatMap => {
                            if let Option::None = self.surface_3d {
                                panic!("No 3d lines initialized, can not make 3d line plot")
                            } else {
                                SupportedPlotTypes::HeatMap
                            }
                        },
                        SupportedPlotTypes::HeatMapAndLines2d => {
                            let surface3d_detected: bool;
                            let lines2d_detected: bool;
                            if let Option::None = self.surface_3d {
                                surface3d_detected = false;
                            } else {
                                surface3d_detected = true;
                            }
                            if let Option::None = self.lines_2d {
                                lines2d_detected = false;
                            } else {
                                lines2d_detected = true;
                            }

                            if (surface3d_detected || lines2d_detected).not() {
                                panic!("No 3d lines initialized, can not make 3d line plot")
                            } else {
                                SupportedPlotTypes::HeatMapAndLines2d
                            }
                        }
                        }
                } else {

                    // get bool of each
                    let plot_2d_line_included: bool;
                    if let Some(_) = self.lines_2d {
                        plot_2d_line_included = true;
                    } else {
                        plot_2d_line_included = false;
                    }

                    let plot_3d_line_included: bool;
                    if let Some(_) = self.lines_3d {
                        plot_3d_line_included = true;
                    } else {
                        plot_3d_line_included = false;
                    }

                    let plot_surface_3d_included: bool;
                    if let Some(_) = self.surface_3d {
                        plot_surface_3d_included = true;
                    } else {
                        plot_surface_3d_included = false;
                    }

                    // match the supported combinations
                    match (plot_2d_line_included, plot_3d_line_included, plot_surface_3d_included) {
                        (true, false, false) => {                       
                            SupportedPlotTypes::Lines2d
                        },
                        (false, true, false) => {
                            SupportedPlotTypes::Lines3d
                        }
                        (false, false, true) => {
                            SupportedPlotTypes::HeatMap
                        }
                        (true, false, true) => {
                            SupportedPlotTypes::HeatMapAndLines2d
                        }
                        (true, true, false) => { // AKA all other options, meant to be _ => { but left as example
                            println!("Found combination of plots: plot_2d_line = {}, plot_3d_line = {}", plot_2d_line_included, plot_3d_line_included);
                            panic!("This combination of plots is not supported!");
                        }, 
                        (false, false, false) => { // AKA all other options, meant to be _ => { but left as example
                            println!("Found combination of plots: plot_2d_line = {}, plot_3d_line = {}", plot_2d_line_included, plot_3d_line_included);
                            panic!("This combination of plots is not supported!");
                        }, 
                        _ => {panic!("This combination of plots is not supported!")}
                    }
                }
            }
        }
    };
}


impl_combined_plots_functions_per_type!(i8);
impl_combined_plots_functions_per_type!(i16);
impl_combined_plots_functions_per_type!(i32);
impl_combined_plots_functions_per_type!(i64);

impl_combined_plots_functions_per_type!(isize);

impl_combined_plots_functions_per_type!(f32);
impl_combined_plots_functions_per_type!(f64);

impl_combined_plots_known_range_functions_per_type!(i8);
impl_combined_plots_known_range_functions_per_type!(i16);
impl_combined_plots_known_range_functions_per_type!(i32);
impl_combined_plots_known_range_functions_per_type!(i64);

impl_combined_plots_known_range_functions_per_type!(isize);

impl_combined_plots_known_range_functions_per_type!(f32);
impl_combined_plots_known_range_functions_per_type!(f64);