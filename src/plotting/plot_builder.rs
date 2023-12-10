use crate::plotting::{*};


#[derive(Clone)]
pub struct PlotBuilder<T> {
    pub lines_2d: Option<Vec<Line2d<T>>>,
    pub lines_3d: Option<Vec<Line3d<T>>>,
    pub surface_3d: Option<Surface3d<T, SurfacePlot>>,
    pub contour: Option<Surface3d<T, ContourPlot>>,
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
    Contour,
    ContourAndLines2d,
    HeatMapContourAndLines2d,
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
    fn scale_plot(&mut self, scale: f64) -> &mut Self;
    fn set_plot_type(&mut self, plot_type: SupportedPlotTypes) -> &mut Self;
    fn guarantee_2d_lines_initialized(&mut self) -> &mut Self;
    fn add_2d_line(&mut self, line_2d: &Line2d<T>) -> &mut Self;
    fn add_simple_2d_line(&mut self, x_data: &Vec<T>, y_data: &Vec<T>) -> &mut Self;
    fn add_simple_2d_line_y_only(&mut self, y_data: &Vec<T>) -> &mut Self;
    fn guarantee_3d_lines_initialized(&mut self) -> &mut Self;
    fn add_3d_line(&mut self, line_3d: &Line3d<T>) -> &mut Self;
    fn add_simple_3d_line(&mut self, x_data: &Vec<T>, y_data: &Vec<T>, z_data: &Vec<T>) -> &mut Self;
    fn add_surface_3d(&mut self, surface_3d: Surface3d<T, SurfacePlot>) -> &mut Self;
    fn add_surface_plot_fn(&mut self, z_function: Rc<Box<dyn Fn(Vec<f64>) -> f64>>) -> &mut Self;
    fn add_surface_plot_xyz(&mut self, x_data: &Vec<T>, y_data: &Vec<T>, z_data: &Vec<Vec<T>>) -> &mut Self;
    fn add_contour(&mut self, contour: Surface3d<T, ContourPlot>) -> &mut Self;
    fn add_contour_plot_fn(&mut self, z_function: Rc<Box<dyn Fn(Vec<f64>) -> f64>>) -> &mut Self;
    fn add_contour_plot_xyz(&mut self, x_data: &Vec<T>, y_data: &Vec<T>, z_data: &Vec<Vec<T>>) -> &mut Self;
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
                    contour: Option::None,
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

            fn scale_plot(&mut self, scale: f64) -> &mut Self {
                match (self.plot_width, self.plot_width) {
                    (Option::Some(width),Option::Some(height)) => {
                        self.plot_width = Option::Some((width as f64 * scale) as u32) ;
                        self.plot_height = Option::Some((height as f64 * scale) as u32) ;
                    },
                    _ => {}
                }
                self.plotting_settings.plot_width = (self.plotting_settings.plot_width as f64 * scale) as u32;
                self.plotting_settings.plot_height = (self.plotting_settings.plot_height as f64 * scale) as u32;
                self.plotting_settings.line_width = (self.plotting_settings.line_width as f64 * scale) as u32;
                self.plotting_settings.outer_figure_margins = (self.plotting_settings.outer_figure_margins as f64 * scale) as u32;
                self.plotting_settings.title_font_size = (self.plotting_settings.title_font_size as f64 * scale) as u32;
                self.plotting_settings.label_font_size = (self.plotting_settings.label_font_size as f64 * scale) as u32;
                self.plotting_settings.x_label_offset = (self.plotting_settings.x_label_offset as f64 * scale) as i32;
                self.plotting_settings.y_label_offset = (self.plotting_settings.y_label_offset as f64 * scale) as i32;
                self.plotting_settings.x_tick_mark_size = (self.plotting_settings.x_tick_mark_size as f64 * scale) as i32;
                self.plotting_settings.y_tick_mark_size = (self.plotting_settings.y_tick_mark_size as f64 * scale) as i32;

                self.plotting_settings.plotters_x_label_area_size = (self.plotting_settings.plotters_x_label_area_size as f64 * scale) as i32;
                self.plotting_settings.plotters_x_top_label_area_size = (self.plotting_settings.plotters_x_top_label_area_size as f64 * scale) as i32;
                self.plotting_settings.plotters_y_label_area_size = (self.plotting_settings.plotters_y_label_area_size as f64 * scale) as i32;
                self.plotting_settings.plotters_right_y_label_area_size = (self.plotting_settings.plotters_right_y_label_area_size as f64 * scale) as i32;
                self.plotting_settings.plotters_margin = (self.plotting_settings.plotters_margin as f64 * scale) as i32;
                self.plotting_settings.plotters_figure_padding = (self.plotting_settings.plotters_figure_padding as f64 * scale) as i32;

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

            fn add_surface_3d(&mut self, surface_3d: Surface3d<$T, SurfacePlot>) -> &mut Self {
                self.surface_3d = Option::Some(surface_3d);

                self
            }

            fn add_surface_plot_fn(&mut self, z_function: Rc<Box<dyn Fn(Vec<f64>) -> f64>>) -> &mut Self {
                let surface_plot = Surface3d::new_surface_fn(z_function);
                self.add_surface_3d(surface_plot);
                self
            }

            fn add_surface_plot_xyz(&mut self, x_data: &Vec<$T>, y_data: &Vec<$T>, z_data: &Vec<Vec<$T>>) -> &mut Self {
                let surface_plot = Surface3d::new_surface_xyz(x_data, y_data, z_data);
                self.add_surface_3d(surface_plot);
                self
            }

            fn add_contour(&mut self, contour: Surface3d<$T, ContourPlot>) -> &mut Self {
                self.contour = Option::Some(contour);

                self
            }

            fn add_contour_plot_fn(&mut self, z_function: Rc<Box<dyn Fn(Vec<f64>) -> f64>>) -> &mut Self {
                let contour_plot = Surface3d::new_contour_fn(z_function);
                self.add_contour(contour_plot);
                self
            }

            fn add_contour_plot_xyz(&mut self, x_data: &Vec<$T>, y_data: &Vec<$T>, z_data: &Vec<Vec<$T>>) -> &mut Self {
                let contour_plot = Surface3d::new_contour_xyz(x_data, y_data, z_data);
                self.add_contour(contour_plot);
                self
            }

            fn to_plot_processor_unitialized(mut self) -> PlotProcessor<$T, NoPlotBackend> {
                self.deduce_ranges_and_axis_types();
                // self.filter_out_of_frame_values();
                PlotProcessor::new_unitialized(self)
            }

            fn to_plotters_processor(mut self) -> PlotProcessor<$T, PlottersBackend> {
                self.deduce_ranges_and_axis_types();
                // self.filter_out_of_frame_values();
                PlotProcessor::new_plotters_backend(self)
            }

            fn to_plotpy_processor(mut self) -> PlotProcessor<$T, PlotPyBackend> {
                self.deduce_ranges_and_axis_types();
                // self.filter_out_of_frame_values();
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
    fn filter_out_of_frame_values(&mut self) -> &mut Self;
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
                    match (&surface_3d.x_values, &surface_3d.y_values, &surface_3d.z_values){
                    (Option::Some(x_values), Option::Some(y_values), Option::Some(z_values)) => {
                        found_any_on_z = true;
                            for x_value in x_values.iter() {
                                if *x_value < x_min {
                                    x_min = *x_value
                                }
                                if *x_value > x_max {
                                    x_max = *x_value
                                }
                            }

                            for y_value in y_values.iter() {
                                if *y_value < y_min {
                                    y_min = *y_value
                                }
                                if *y_value > y_max {
                                    y_max = *y_value
                                }
                            }

                            for z_value_row in z_values.iter() {
                                for z_value in z_value_row {
                                    if *z_value < z_min {
                                        z_min = *z_value
                                    }
                                    if *z_value > z_max {
                                        z_max = *z_value
                                    }
                                }
                            }
                        },
                        _ => {}

                    }

                } 
                if let Option::Some(contour) = &self.contour { 
                    match (&contour.x_values, &contour.y_values, &contour.z_values){
                    (Option::Some(x_values), Option::Some(y_values), Option::Some(z_values)) => {
                        found_any_on_z = true;
                            for x_value in x_values.iter() {
                                if *x_value < x_min {
                                    x_min = *x_value
                                }
                                if *x_value > x_max {
                                    x_max = *x_value
                                }
                            }

                            for y_value in y_values.iter() {
                                if *y_value < y_min {
                                    y_min = *y_value
                                }
                                if *y_value > y_max {
                                    y_max = *y_value
                                }
                            }

                            for z_value_row in z_values.iter() {
                                for z_value in z_value_row {
                                    if *z_value < z_min {
                                        z_min = *z_value
                                    }
                                    if *z_value > z_max {
                                        z_max = *z_value
                                    }
                                }
                            }
                        },
                        _ => {}

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

            fn filter_out_of_frame_values(&mut self) -> &mut Self {
                let x_range = self.x_range.clone().unwrap();
                let y_range = self.y_range.clone().unwrap();
                let y2_range = self.y2_range.clone().unwrap();

                let lines_2d_found: bool;
                let mut lines: Vec<Line2d<$T>>;
                match &self.lines_2d {
                    Option::Some(lines_2d) => {
                        lines_2d_found = true;
                        lines = lines_2d.clone()
                    },
                    Option::None => {
                        lines_2d_found = false;
                        lines = Vec::<Line2d<$T>>::new();
                    }
                }

                if lines_2d_found {
                    for i_l in 0..lines.len() {
                        let relavant_y_range: Range<$T>;
                        if lines[i_l].on_secondary_axis {
                            relavant_y_range = y2_range.clone();
                        } else {
                            relavant_y_range = y_range.clone();
                        }

                        for i_p in 0.. lines[i_l].x_values.len()-1 {

                            let is_in_x_range = x_range.contains(&lines[i_l].x_values[i_p]);
                            let is_in_y_range = relavant_y_range.contains(&lines[i_l].x_values[i_p]);

                            if !(is_in_x_range && is_in_y_range) {
                                lines[i_l].x_values[i_p] = $T::INFINITY;
                                lines[i_l].y_values[i_p] = $T::INFINITY;
                            }        
                    
                        
                        }
                    }
                    self.lines_2d = Option::Some(lines);
                }

                


                // // "find_point_that_intersects_boundry" breaks if the ranges are reversed!
                // assert!(self.x_range.as_ref().unwrap().end > self.x_range.as_ref().unwrap().start);
                // assert!(self.y_range.as_ref().unwrap().end > self.y_range.as_ref().unwrap().start);

                // fn check_neighbour_is_in_range(lines: &Vec<Line2d<$T>>, i_l: usize, i_p: usize, x_range: &Range<$T>, y_range: &Range<$T>) -> bool {
                //     let is_in_x_range = x_range.contains(&lines[i_l].x_values[i_p]);
                //     let is_in_y_range = y_range.contains(&lines[i_l].x_values[i_p]);

                //     if (is_in_x_range && is_in_y_range) {
                //         true
                //     } else {
                //         false
                //     }
                // }

                // fn find_point_that_intersects_boundry(lines: &Vec<Line2d<$T>>, i_l: usize, i_p: usize, i_p_other: usize, x_range: &Range<$T>, y_range: &Range<$T>) -> ($T, $T) {
                //     // this function breaks if range is reversed
                //     let x = lines[i_l].x_values[i_p];
                //     let y = lines[i_l].y_values[i_p];
                    
                //     let x_other = lines[i_l].x_values[i_p_other];
                //     let y_other = lines[i_l].y_values[i_p_other];

                //     let a = (y - y_other) / (x - x_other);
                //     let b = y - a*x;

                //     let x_intersect_y_upper = (y_range.end - b) / a;
                //     let x_intersect_y_lower = (y_range.start - b) / a;

                //     let y_intersect_x_left = a*x_range.start + b;
                //     let y_intersect_x_right = a*x_range.end + b;

                //     if a > 0. {
                //         if a * x_range.start + b < y_range.start {
                //             let length_to_intersect_x_right = ( (x - x_range.end).powi(2) + (y - y_intersect_x_right).powi(2) ).sqrt();
                //             let length_to_intersect_y_low = ( (x - x_intersect_y_lower).powi(2) + (y - y_range.start).powi(2) ).sqrt();
                //             if length_to_intersect_x_right < length_to_intersect_y_low {
                //                 (x_range.end, y_intersect_x_right)
                //             } else {
                //                 (x_intersect_y_lower, y_range.start)
                //             }
                //         } else {
                //             let length_to_intersect_x_left = ( (x - x_range.start).powi(2) + (y - y_intersect_x_left).powi(2) ).sqrt();
                //             let length_to_intersect_y_upper = ( (x - x_intersect_y_upper).powi(2) + (y - y_range.end).powi(2) ).sqrt();
                //             if length_to_intersect_x_left < length_to_intersect_y_upper {
                //                 (x_range.start, y_intersect_x_left)
                //             } else {
                //                 (x_intersect_y_upper, y_range.end)
                //             }
                //         }
                //     } else {
                //         if a * x_range.start + b < y_range.end {
                //             let length_to_intersect_x_left = ( (x - x_range.start).powi(2) + (y - y_intersect_x_left).powi(2) ).sqrt();
                //             let length_to_intersect_y_low = ( (x - x_intersect_y_lower).powi(2) + (y - y_range.start).powi(2) ).sqrt();
                //             if length_to_intersect_x_left < length_to_intersect_y_low {
                //                 (x_range.start, y_intersect_x_left)
                //             } else {
                //                 (x_intersect_y_lower, y_range.start)
                //             }
                //         } else {
                //             let length_to_intersect_y_upper = ( (x - x_intersect_y_upper).powi(2) + (y - y_range.end).powi(2) ).sqrt();
                //             let length_to_intersect_x_right = ( (x - x_range.end).powi(2) + (y - y_intersect_x_right).powi(2) ).sqrt();
                //             if length_to_intersect_y_upper < length_to_intersect_x_right {
                //                 (x_intersect_y_upper, y_range.end)
                //             } else {
                //                 (x_range.end, y_intersect_x_right)
                //             }
                //         }
                //     }
                // }

                // let x_range = self.x_range.clone().unwrap();
                // let y_range = self.y_range.clone().unwrap();
                // let y2_range = self.y2_range.clone().unwrap();

                // let lines_2d_corrected: Option<Vec<Line2d<$T>>>;
                // match &self.lines_2d {
                //     Option::Some(lines_2d) => {
                //         let mut lines = lines_2d.clone();
                //         for i_l in 0..lines.len() {


                //             let relavant_y_range: Range<$T>;
                //             if lines[i_l].on_secondary_axis {
                //                 relavant_y_range = y2_range.clone();
                //             } else {
                //                 relavant_y_range = y_range.clone();
                //             }

                //             let mut new_x_values = Vec::<$T>::with_capacity(lines[i_l].x_values.len());
                //             let mut new_y_values = Vec::<$T>::with_capacity(lines[i_l].x_values.len());

                //             for i_p in 0.. lines[i_l].x_values.len()-1 {

                //                 let is_in_x_range = x_range.contains(&lines[i_l].x_values[i_p]);
                //                 let is_in_y_range = relavant_y_range.contains(&lines[i_l].x_values[i_p]);

                //                 let self_is_in_range: bool;
                //                 if is_in_x_range && is_in_y_range {
                //                     self_is_in_range = true;
                //                     new_x_values.push(lines[i_l].x_values[i_p]);
                //                     new_y_values.push(lines[i_l].x_values[i_p]);
                //                 } else {
                //                     self_is_in_range = false;
                //                 }

                //                 let right_neigbour_is_in_range = check_neighbour_is_in_range(&lines, i_l, i_p+1, &x_range, &y_range);

                //                 match (self_is_in_range, right_neigbour_is_in_range) {
                //                     (true, true) => {}, // will be added normally
                //                     (true, false) => { // add point on intersect with frame
                //                         let (x_inter, y_inter) = find_point_that_intersects_boundry(&lines, i_l, i_p, i_p+1, &x_range, &y_range);
                //                         new_x_values.push(x_inter);
                //                         new_y_values.push(y_inter);
                //                     }
                //                     (false, true) => { // add nan to disconnect points, insert point on edge boundry
                //                         new_x_values.push($T::NAN);
                //                         new_y_values.push($T::NAN);
                //                         let (x_inter, y_inter) = find_point_that_intersects_boundry(&lines, i_l, i_p, i_p+1, &x_range, &y_range);
                //                         new_x_values.push(x_inter);
                //                         new_y_values.push(y_inter);
                //                     }
                //                     (false, false) => {} // no points need to be added.
                //                 }
                                
                //             }

                //             lines[i_l].x_values = new_x_values;
                //             lines[i_l].y_values = new_y_values;
                //         }
                //         lines_2d_corrected = Option::Some(lines);
                //     },
                //     Option::None => {lines_2d_corrected = Option::None}
                // }

                // self.lines_2d = lines_2d_corrected;

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
                                panic!("No surface initialized, can not make surface plot")
                            } else {
                                SupportedPlotTypes::HeatMap
                            }
                        },
                        SupportedPlotTypes::Contour => {
                            if let Option::None = self.surface_3d {
                                panic!("No contour plot initialized, can not make contour plot")
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
                                panic!("No surface initialized, can not make heatmap / line plot")
                            } else {
                                SupportedPlotTypes::HeatMapAndLines2d
                            }
                        },
                        SupportedPlotTypes::ContourAndLines2d => {
                            let contour_detected: bool;
                            let lines2d_detected: bool;
                            if let Option::None = self.contour {
                                contour_detected = false;
                            } else {
                                contour_detected = true;
                            }
                            if let Option::None = self.lines_2d {
                                lines2d_detected = false;
                            } else {
                                lines2d_detected = true;
                            }

                            if (contour_detected || lines2d_detected).not() {
                                panic!("No contour initialized, can not make contour / line line plot")
                            } else {
                                SupportedPlotTypes::ContourAndLines2d
                            }
                        },
                        SupportedPlotTypes::HeatMapContourAndLines2d => {
                            let surface3d_detected: bool;
                            let contour_detected: bool;
                            let lines2d_detected: bool;
                            if let Option::None = self.surface_3d {
                                surface3d_detected = false;
                            } else {
                                surface3d_detected = true;
                            }
                            if let Option::None = self.contour {
                                contour_detected = false;
                            } else {
                                contour_detected = true;
                            }
                            if let Option::None = self.lines_2d {
                                lines2d_detected = false;
                            } else {
                                lines2d_detected = true;
                            }

                            if (surface3d_detected || contour_detected || lines2d_detected).not() {
                                panic!("No contour initialized, can not make contour / line line plot")
                            } else {
                                SupportedPlotTypes::ContourAndLines2d
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
                    let plot_contour_included: bool;
                    if let Some(_) = self.contour {
                        plot_contour_included = true;
                    } else {
                        plot_contour_included = false;
                    }


                    // match the supported combinations
                    match (plot_2d_line_included, plot_3d_line_included, plot_surface_3d_included,plot_contour_included) {
                        (true, false, false, false) => {                       
                            SupportedPlotTypes::Lines2d
                        },
                        (false, true, false, false) => {
                            SupportedPlotTypes::Lines3d
                        }
                        (false, false, true, false) => {
                            SupportedPlotTypes::HeatMap
                        }
                        (true, false, true, false) => {
                            SupportedPlotTypes::HeatMapAndLines2d
                        }
                        (false, false, false, true) => {
                            SupportedPlotTypes::Contour
                        }
                        (true, false, false, true) => {
                            SupportedPlotTypes::ContourAndLines2d
                        }
                        (true, false, true, true) => {
                            SupportedPlotTypes::HeatMapContourAndLines2d
                        }
                        // (true, true, false, false) => { // AKA all other options, meant to be _ => { but left as example
                        //     println!("Found combination of plots: plot_2d_line = {}, plot_3d_line = {}", plot_2d_line_included, plot_3d_line_included);
                        //     panic!("This combination of plots is not supported!");
                        // }, 
                        // (false, false, false) => { // AKA all other options, meant to be _ => { but left as example
                        //     println!("Found combination of plots: plot_2d_line = {}, plot_3d_line = {}", plot_2d_line_included, plot_3d_line_included);
                        //     panic!("This combination of plots is not supported!");
                        // }, 
                        _ => {panic!("This combination of plots is not supported!")}
                    }
                }
            }
        }
    };
}




impl_combined_plots_functions_per_type!(f32);
impl_combined_plots_functions_per_type!(f64);


impl_combined_plots_known_range_functions_per_type!(f32);
impl_combined_plots_known_range_functions_per_type!(f64);