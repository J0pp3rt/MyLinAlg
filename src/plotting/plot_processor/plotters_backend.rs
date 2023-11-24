use crate::plotting::{*};

pub trait PlotProcessorPlottersBackendFunctions<T> {

    fn SVG_to_file(&self, file_path: &str);
    fn SVG_to_mem(&self) -> String;
    fn SVG_to_RGBA(&self) -> Box<[u8]>;

}
pub trait PlotProcessorPlottersBackendFunctionsForAllThePlottersBackends<OutputType: plotters::prelude::DrawingBackend> {
    fn process(&self, root: &mut DrawingArea<OutputType, Shift>);
    fn simple_2d_plot(&self, root: &mut DrawingArea<OutputType, Shift>);
    fn simple_3d_plot(&self, root: &mut DrawingArea<OutputType, Shift>);
}

macro_rules! impl_plot_processor_plotters_backend_functions_per_type_with_annoying_variants {
    ($T: ident) => { 
        impl<OutputType: plotters::prelude::DrawingBackend> PlotProcessorPlottersBackendFunctionsForAllThePlottersBackends<OutputType> for PlotProcessor<$T, PlottersBackend> {
        fn process(&self, root: &mut DrawingArea<OutputType, Shift>) {

            // Some standard settings
            root.margin(self.plots.plotting_settings.plotters_figure_padding, self.plots.plotting_settings.plotters_figure_padding, self.plots.plotting_settings.plotters_figure_padding,self.plots.plotting_settings.plotters_figure_padding)
                .fill(&WHITE).unwrap();

            match (self.plots.indentify_plot_type() ){
                SupportedPlotTypes::Lines2d => {
                    self.simple_2d_plot(root)
                },
                SupportedPlotTypes::Lines3d => {
                    self.simple_3d_plot(root)
                },
                // _ => {panic!("This plot type is not supported by the PLOTTERS backend")}
            }

            // indentify the kind of plot

            // do some standard things

            // add feature for all the things that are in there

            // call .present() to make it all happen
            root.present().unwrap();

            // done
        }

        fn simple_2d_plot(&self, root: &mut DrawingArea<OutputType, Shift>) {

            let title_of_figure: String;
            match &self.plots.title.is_empty() {
                false => {title_of_figure = self.plots.title.clone()}
                true => {title_of_figure = self.plots.plotting_settings.title.clone()}
            }

            let range_x_f64 = self.plots.x_range.as_ref().unwrap().start as f64 .. self.plots.x_range.as_ref().unwrap().end as f64;
            let range_y_f64 = self.plots.y_range.as_ref().unwrap().start as f64 .. self.plots.y_range.as_ref().unwrap().end as f64;
            let range_y2_f64 = self.plots.y2_range.as_ref().unwrap().start as f64 .. self.plots.y2_range.as_ref().unwrap().end as f64;
            // let range_y2_f64_dummy = 1. as f64 .. 1.1 as f64;

            // using unwrap 
            let x_axis_scaling = self.plots.x_axis_scaling.clone().expect("x axis scaling should have been set!");
            let y_axis_scaling = self.plots.y_axis_scaling.clone().expect("y axis scaling should have been set!");
            let y2_axis_scaling = self.plots.y2_axis_scaling.clone().expect("y2 axis scaling should have been set!");

            let mut chart = ChartBuilder::on(&root);
            chart
            .x_label_area_size(self.plots.plotting_settings.plotters_x_label_area_size)
            .y_label_area_size(self.plots.plotting_settings.plotters_y_label_area_size)
            .right_y_label_area_size(self.plots.plotting_settings.plotters_right_y_label_area_size)
            .margin(self.plots.plotting_settings.plotters_margin)
            .caption(title_of_figure, ("sans-serif", 25.).into_font());

            // get ready for some an-idiometicness
            match (x_axis_scaling, y_axis_scaling, y2_axis_scaling) {
                (PlotAxisScaling::Linear, PlotAxisScaling::Linear, PlotAxisScaling::NoAxis) => {
                    let mut chart = chart
                        .build_cartesian_2d(range_x_f64.clone(), range_y_f64).unwrap()
                        .set_secondary_coord(range_x_f64, range_y2_f64);
                    
                    produce_other_2d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Log, PlotAxisScaling::Linear, PlotAxisScaling::NoAxis) => {
                    let mut chart = chart
                        .build_cartesian_2d((range_x_f64.clone()).log_scale(), range_y_f64).unwrap()
                        .set_secondary_coord(range_x_f64, range_y2_f64);
                    
                    produce_other_2d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Linear, PlotAxisScaling::Log, PlotAxisScaling::NoAxis) => {
                    let mut chart = chart
                        .build_cartesian_2d(range_x_f64.clone(), (range_y_f64).log_scale()).unwrap()
                        .set_secondary_coord(range_x_f64, range_y2_f64);
                    
                    produce_other_2d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Log, PlotAxisScaling::Log, PlotAxisScaling::NoAxis) => {
                    let mut chart = chart
                        .build_cartesian_2d((range_x_f64.clone()).log_scale(), (range_y_f64).log_scale()).unwrap()
                        .set_secondary_coord(range_x_f64, range_y2_f64);

                    produce_other_2d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Linear, PlotAxisScaling::Linear, PlotAxisScaling::Linear) => {
                    let mut chart = chart
                        .build_cartesian_2d(range_x_f64.clone(), range_y_f64).unwrap()
                        .set_secondary_coord(range_x_f64, range_y2_f64);

                    produce_other_2d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Linear, PlotAxisScaling::Linear, PlotAxisScaling::Log) => {
                    let mut chart = chart
                        .build_cartesian_2d(range_x_f64.clone(), range_y_f64).unwrap()
                        .set_secondary_coord(range_x_f64, (range_y2_f64).log_scale());
                    produce_other_2d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Log, PlotAxisScaling::Linear, PlotAxisScaling::Linear) => {
                    let mut chart = chart
                        .build_cartesian_2d((range_x_f64.clone()).log_scale(), range_y_f64).unwrap()
                        .set_secondary_coord((range_x_f64).log_scale(), range_y2_f64);

                    produce_other_2d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Log, PlotAxisScaling::Linear, PlotAxisScaling::Log) => {
                    let mut chart = chart
                        .build_cartesian_2d((range_x_f64.clone()).log_scale(), range_y_f64).unwrap()
                        .set_secondary_coord((range_x_f64).log_scale(), (range_y2_f64).log_scale());

                    produce_other_2d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Linear, PlotAxisScaling::Log, PlotAxisScaling::Linear) => {
                    let mut chart = chart
                        .build_cartesian_2d(range_x_f64.clone(), (range_y_f64).log_scale()).unwrap()
                        .set_secondary_coord(range_x_f64, range_y2_f64);

                    produce_other_2d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Linear, PlotAxisScaling::Log, PlotAxisScaling::Log) => {
                    let mut chart = chart
                        .build_cartesian_2d(range_x_f64.clone(), (range_y_f64).log_scale()).unwrap()
                        .set_secondary_coord(range_x_f64, (range_y2_f64).log_scale());

                    produce_other_2d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Log, PlotAxisScaling::Log, PlotAxisScaling::Linear) => {
                    let mut chart = chart
                        .build_cartesian_2d((range_x_f64.clone()).log_scale(), (range_y_f64).log_scale()).unwrap()
                        .set_secondary_coord((range_x_f64).log_scale(), range_y2_f64);

                    produce_other_2d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Log, PlotAxisScaling::Log, PlotAxisScaling::Log) => {
                    let mut chart = chart
                        .build_cartesian_2d((range_x_f64.clone()).log_scale(), (range_y_f64).log_scale()).unwrap()
                        .set_secondary_coord((range_x_f64).log_scale(), (range_y2_f64).log_scale());

                    produce_other_2d_plot_settings!(chart, self);
                },
                _ => {panic!("Either there is a genuine axis combination missing or something is messed up! :(")}
            }
        }

        fn simple_3d_plot(&self, root: &mut DrawingArea<OutputType, Shift>) {

            let title_of_figure: String;
            match &self.plots.title.is_empty() {
                false => {title_of_figure = self.plots.title.clone()}
                true => {title_of_figure = self.plots.plotting_settings.title.clone()}
            }

            let range_x_f64 = self.plots.x_range.as_ref().unwrap().start as f64 .. self.plots.x_range.as_ref().unwrap().end as f64;
            let range_y_f64 = self.plots.y_range.as_ref().unwrap().start as f64 .. self.plots.y_range.as_ref().unwrap().end as f64;
            let range_z_f64 = self.plots.z_range.as_ref().unwrap().start as f64 .. self.plots.z_range.as_ref().unwrap().end as f64;

            // using unwrap 
            let x_axis_scaling = self.plots.x_axis_scaling.clone().expect("x axis scaling should have been set!");
            let y_axis_scaling = self.plots.y_axis_scaling.clone().expect("y axis scaling should have been set!");
            let z_axis_scaling = self.plots.z_axis_scaling.clone().expect("z axis scaling should have been set!");

            let mut chart = ChartBuilder::on(&root);
            chart
            .x_label_area_size(self.plots.plotting_settings.plotters_x_label_area_size)
            .y_label_area_size(self.plots.plotting_settings.plotters_y_label_area_size)
            .right_y_label_area_size(self.plots.plotting_settings.plotters_right_y_label_area_size)
            .margin(self.plots.plotting_settings.plotters_margin)
            .caption(title_of_figure, ("sans-serif", 25.).into_font());

            // get ready for some an-idiometicness
            match (x_axis_scaling, y_axis_scaling, z_axis_scaling) {
                (PlotAxisScaling::Linear, PlotAxisScaling::Linear, PlotAxisScaling::Linear) => {
                    let mut chart = chart
                        .build_cartesian_3d(range_x_f64, range_y_f64, range_z_f64).unwrap();
                    
                    produce_other_3d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Log, PlotAxisScaling::Linear, PlotAxisScaling::Linear) => {
                    let mut chart = chart
                        .build_cartesian_3d(range_x_f64.log_scale(), range_y_f64, range_z_f64).unwrap();
                    
                    produce_other_3d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Linear, PlotAxisScaling::Log, PlotAxisScaling::Linear) => {
                    let mut chart = chart
                        .build_cartesian_3d(range_x_f64, range_y_f64.log_scale(), range_z_f64).unwrap();
                    
                    produce_other_3d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Linear, PlotAxisScaling::Linear, PlotAxisScaling::Log) => {
                    let mut chart = chart
                    .build_cartesian_3d(range_x_f64, range_y_f64, range_z_f64.log_scale()).unwrap();

                    produce_other_3d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Log, PlotAxisScaling::Log, PlotAxisScaling::Linear) => {
                    let mut chart = chart
                    .build_cartesian_3d(range_x_f64.log_scale(), range_y_f64.log_scale(), range_z_f64).unwrap();

                    produce_other_3d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Log, PlotAxisScaling::Linear, PlotAxisScaling::Log) => {
                    let mut chart = chart
                    .build_cartesian_3d(range_x_f64.log_scale(), range_y_f64, range_z_f64.log_scale()).unwrap();

                    produce_other_3d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Linear, PlotAxisScaling::Log, PlotAxisScaling::Log) => {
                    let mut chart = chart
                    .build_cartesian_3d(range_x_f64, range_y_f64.log_scale(), range_z_f64.log_scale()).unwrap();

                    produce_other_3d_plot_settings!(chart, self);
                },
                (PlotAxisScaling::Log, PlotAxisScaling::Log, PlotAxisScaling::Log) => {
                    let mut chart = chart
                    .build_cartesian_3d(range_x_f64.log_scale(), range_y_f64.log_scale(), range_z_f64.log_scale()).unwrap();

                    produce_other_3d_plot_settings!(chart, self);
                },
                _ => {panic!("Either there is a genuine axis combination missing or something is messed up! :(")}
            }
        }
    }
}}


macro_rules! impl_plot_processor_plotters_backend_functions_per_type {
    ($T: ident) => {
        impl PlotProcessorPlottersBackendFunctions<$T> for PlotProcessor<$T, PlottersBackend> {
        
            fn SVG_to_file(&self, file_path: &str) {
                let (width, height) = self.plots.get_plot_dimensions();

                let mut root = SVGBackend::new(file_path, (width, height)).into_drawing_area();

                self.process(&mut root);
            }

            fn SVG_to_mem(&self) -> String {
                let (width, height) = self.plots.get_plot_dimensions(); 

                let mut string_buffer = String::new();
                {
                    let mut root = SVGBackend::with_string(&mut string_buffer, (width, height)).into_drawing_area();
                    self.process(&mut root);
                }
                string_buffer
            }
            
            fn SVG_to_RGBA(&self) -> Box<[u8]> {
                // probably this one utilizes SVG_to_mem first
                todo!()
            }

            
        }
    }
}




macro_rules! produce_other_2d_plot_settings {
    ($chart: expr, $self: expr) => {
        let mut axis = $chart.configure_mesh();

        match $self.plots.plotting_settings.x_grid_major_subdevisions {
            Option::Some(n_lines) => {
                axis.x_labels(n_lines);
            },
            _ => {} // default is to already show major gridlines
        }

        match $self.plots.plotting_settings.y_grid_major_subdevisions {
            Option::Some(n_lines) => {
                axis.y_labels(n_lines);
            },
            _ => {} // default is to already show major gridlines
        }

        match $self.plots.plotting_settings.show_x_grid_minor {
            true => {
                match $self.plots.plotting_settings.x_grid_minor_subdevisions {
                    Option::Some(n_lines) => {
                        axis.x_max_light_lines(n_lines);
                    },
                    _ => {} // default is to already show minor gridlines
                }
            },
            false => {
                axis.x_max_light_lines(0);
            }
        }

        match $self.plots.plotting_settings.show_y_grid_minor {
            true => {
                match $self.plots.plotting_settings.y_grid_minor_subdevisions {
                    Option::Some(n_lines) => {
                        axis.y_max_light_lines(n_lines);
                    },
                    _ => {} // default is to already show minor gridlines
                }
            },
            false => {
                axis.y_max_light_lines(0);
            }
        }

        match $self.plots.plotting_settings.show_x_grid_major {
            false => {
                println!("Disabling major gridlines not supported yet...");
            },
            true => {}
        }

        match $self.plots.plotting_settings.show_y_grid_major {
            false => {
                println!("Disabling major gridlines not supported yet...");
            },
            true => {}
        }

        match $self.plots.plotting_settings.show_x_axis {
            false => {
                axis.disable_x_axis();
            },
            _ => {}
        }

        match $self.plots.plotting_settings.show_y_axis {
            false => {
                axis.disable_y_axis();
            },
            _ => {}
        }

        match $self.plots.plotting_settings.show_x_mesh {
            false => {
                axis.disable_x_mesh();
            },
            _ => {}
        }

        match $self.plots.plotting_settings.show_y_mesh {
            false => {
                axis.disable_y_mesh();
            },
            _ => {}
        }

        axis
            .x_desc($self.plots.x_label.clone())
            .y_desc($self.plots.y_label.clone())
            .draw().unwrap(); // do some more work on this

        match $self.plots.y2_axis_scaling.as_ref().unwrap() {
            PlotAxisScaling::NoAxis => {},
            _ => {
                // process the settings for the secondary axis, maybe this can be made nicer
                let secondary_axis_label = $self.plots.y2_label.clone();

                $chart
                    .configure_secondary_axes()
                    .y_desc(secondary_axis_label)
                    .draw().unwrap();
            }
        }

        let amount_of_lines = $self.plots.lines_2d.as_ref().unwrap().len();
        for (number, series) in $self.plots.lines_2d.as_ref().unwrap().iter().enumerate() {
            let legend_name: String;
            if !(&series.name.is_empty()) {
                legend_name = series.name.clone();
            } else {
                legend_name = { format!{"Data_series {number}"} }
            }


            let line_width: u32;
            if let Option::Some(prefered_line_width) = series.line_width {
                line_width = prefered_line_width;
            } else {
                line_width = $self.plots.plotting_settings.line_width;
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
                match $self.plots.plotting_settings.marker_style {
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
                match $self.plots.plotting_settings.marker_fill {
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
                if let Option::Some(color) = $self.plots.plotting_settings.line_color {
                    line_color = color;
                } else {
                    line_color = $self.plots.plotting_settings.color_map.get_color(number as f32/ amount_of_lines as f32);
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

macro_rules! produce_other_3d_plot_settings {
    ($chart: expr, $self: expr) => {

        $chart.with_projection(|mut projection|{
            projection.pitch = $self.plots.plotting_settings.plot_3d_pitch;
            projection.yaw = $self.plots.plotting_settings.plot_3d_yaw;
            projection.scale = $self.plots.plotting_settings.plot_3d_scale;
            projection.into_matrix()
        });

        let mut axis = $chart.configure_axes();

        match $self.plots.plotting_settings.x_grid_major_subdevisions {
            Option::Some(n_lines) => {
                axis.x_labels(n_lines);
            },
            _ => {} // default is to already show major gridlines
        }

        match $self.plots.plotting_settings.y_grid_major_subdevisions {
            Option::Some(n_lines) => {
                axis.y_labels(n_lines);
            },
            _ => {} // default is to already show major gridlines
        }

        match $self.plots.plotting_settings.show_x_grid_minor {
            true => {
                match $self.plots.plotting_settings.x_grid_minor_subdevisions {
                    Option::Some(n_lines) => {
                        axis.x_max_light_lines(n_lines);
                    },
                    _ => {} // default is to already show minor gridlines
                }
            },
            false => {
                axis.x_max_light_lines(0);
            }
        }

        match $self.plots.plotting_settings.show_y_grid_minor {
            true => {
                match $self.plots.plotting_settings.y_grid_minor_subdevisions {
                    Option::Some(n_lines) => {
                        axis.y_max_light_lines(n_lines);
                    },
                    _ => {} // default is to already show minor gridlines
                }
            },
            false => {
                axis.y_max_light_lines(0);
            }
        }

        match $self.plots.plotting_settings.show_x_grid_major {
            false => {
                println!("Disabling major gridlines not supported yet...");
            },
            true => {}
        }

        match $self.plots.plotting_settings.show_y_grid_major {
            false => {
                println!("Disabling major gridlines not supported yet...");
            },
            true => {}
        }

        // match $self.plots.plotting_settings.show_x_axis {
        //     false => {
        //         axis.disable_x_axis();
        //     },
        //     _ => {}
        // }

        // match $self.plots.plotting_settings.show_y_axis {
        //     false => {
        //         axis.disable_y_axis();
        //     },
        //     _ => {}
        // }

        // match $self.plots.plotting_settings.show_x_mesh {
        //     false => {
        //         axis.disable_x_mesh();
        //     },
        //     _ => {}
        // }

        // match $self.plots.plotting_settings.show_y_mesh {
        //     false => {
        //         axis.disable_y_mesh();
        //     },
        //     _ => {}
        // }

        axis
            .draw().unwrap(); // do some more work on this

        let amount_of_lines = $self.plots.lines_3d.as_ref().unwrap().len();
        for (number, series) in $self.plots.lines_3d.as_ref().unwrap().iter().enumerate() {
            let legend_name: String;
            if !(&series.name.is_empty()) {
                legend_name = series.name.clone();
            } else {
                legend_name = { format!{"Data_series {number}"} }
            }


            let line_width: u32;
            if let Option::Some(prefered_line_width) = series.line_width {
                line_width = prefered_line_width;
            } else {
                line_width = $self.plots.plotting_settings.line_width;
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
                match $self.plots.plotting_settings.marker_style {
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
                match $self.plots.plotting_settings.marker_fill {
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
                if let Option::Some(color) = $self.plots.plotting_settings.line_color {
                    line_color = color;
                } else {
                    line_color = $self.plots.plotting_settings.color_map.get_color(number as f32/ amount_of_lines as f32);
                }
            }

            let style = ShapeStyle {color: line_color, filled: marker_fill, stroke_width: line_width};

            $chart
                // .draw_series(LineSeries::new(series.x_values.iter().zip(series.y_values.clone()).map(|(x, y)| (*x as f64, y as f64)), style)
                .draw_series(LineSeries::new(
                    (0..series.x_values.len()).map(|index|
                    (series.x_values[index] as f64, series.y_values[index] as f64, series.z_values[index] as f64)),
                style)
                .point_size(marker_size))
                .unwrap()
                .label(legend_name);
       
        }
    }
}

impl_plot_processor_plotters_backend_functions_per_type!(i8);
impl_plot_processor_plotters_backend_functions_per_type!(i16);
impl_plot_processor_plotters_backend_functions_per_type!(i32);
impl_plot_processor_plotters_backend_functions_per_type!(i64);

impl_plot_processor_plotters_backend_functions_per_type!(isize);

impl_plot_processor_plotters_backend_functions_per_type!(f32);
impl_plot_processor_plotters_backend_functions_per_type!(f64);

impl_plot_processor_plotters_backend_functions_per_type_with_annoying_variants!(i8);
impl_plot_processor_plotters_backend_functions_per_type_with_annoying_variants!(i16);
impl_plot_processor_plotters_backend_functions_per_type_with_annoying_variants!(i32);
impl_plot_processor_plotters_backend_functions_per_type_with_annoying_variants!(i64);

impl_plot_processor_plotters_backend_functions_per_type_with_annoying_variants!(isize);

impl_plot_processor_plotters_backend_functions_per_type_with_annoying_variants!(f32);
impl_plot_processor_plotters_backend_functions_per_type_with_annoying_variants!(f64);
