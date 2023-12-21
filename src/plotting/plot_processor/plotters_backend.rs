use crate::plotting::{*};
use plotters::style::text_anchor::{HPos, Pos, VPos};

use plotters::prelude::{*};

use crate::full_palette::GREY;
use plotters::chart::DualCoordChartContext;
pub trait PlotProcessorPlottersBackendFunctions<T> {

    fn SVG_to_file(&mut self, file_path: &str);
    fn SVG_to_mem(&mut self) -> String;
    fn SVG_to_RGBA(&mut self) -> Vec<u8>;
    fn bitmap_to_rgb(&mut self) -> Vec<u8>;
    fn bitmap_to_file(&mut self, file_path: &str);
}
pub trait PlotProcessorPlottersBackendFunctionsForAllThePlottersBackends<OutputType: plotters::prelude::DrawingBackend> {
    fn process(&self, root: &mut DrawingArea<OutputType, Shift>);
    fn simple_2d_plot(&self, root: &mut DrawingArea<OutputType, Shift>, is_primary_plot: bool);
    fn simple_3d_plot(&self, root: &mut DrawingArea<OutputType, Shift>, is_primary_plot: bool);
    fn heatmap_plot(&self, root: &mut DrawingArea<OutputType, Shift>, is_primary_plot: bool);
    fn contour_plot(&self, root: &mut DrawingArea<OutputType, Shift>, is_primary_plot: bool);
    fn polygon_plot(&self, root: &mut DrawingArea<OutputType, Shift>, is_primary_plot: bool);
}

macro_rules! make_std_chart {
    ($self: expr, $chart: expr) => {
        $chart
            .x_label_area_size($self.plots.plotting_settings.plotters_x_label_area_size)
            .y_label_area_size($self.plots.plotting_settings.plotters_y_label_area_size)
            .top_x_label_area_size($self.plots.plotting_settings.plotters_x_top_label_area_size)
            .right_y_label_area_size($self.plots.plotting_settings.plotters_right_y_label_area_size)
            .margin($self.plots.plotting_settings.plotters_margin)
            .caption({
                match &$self.plots.title.is_empty() {
                    false => {$self.plots.title.clone()}
                    true => {$self.plots.plotting_settings.title.clone()}
                }
            }, TextStyle::from(("sans-serif", $self.plots.plotting_settings.title_font_size).into_font()).pos(Pos::new(HPos::Center, VPos::Top)))
    }
}

macro_rules! make_std_axis {
    ($axis: expr, $self: expr) => {


        match $self.plots.plotting_settings.x_grid_major_subdevisions {
            Option::Some(n_lines) => {
                $axis.x_labels(n_lines);
            },
            _ => {} // default is to already show major gridlines
        }

        match $self.plots.plotting_settings.y_grid_major_subdevisions {
            Option::Some(n_lines) => {
                $axis.y_labels(n_lines);
            },
            _ => {} // default is to already show major gridlines
        }

        match $self.plots.plotting_settings.show_x_grid_minor {
            true => {
                match $self.plots.plotting_settings.x_grid_minor_subdevisions {
                    Option::Some(n_lines) => {
                        $axis.x_max_light_lines(n_lines);
                    },
                    _ => {} // default is to already show minor gridlines
                }
            },
            false => {
                $axis.x_max_light_lines(0);
            }
        }

        match $self.plots.plotting_settings.show_y_grid_minor {
            true => {
                match $self.plots.plotting_settings.y_grid_minor_subdevisions {
                    Option::Some(n_lines) => {
                        $axis.y_max_light_lines(n_lines);
                    },
                    _ => {} // default is to already show minor gridlines
                }
            },
            false => {
                $axis.y_max_light_lines(0);
            }
        }

        match $self.plots.plotting_settings.show_grid_major {
            false => {
                $axis.bold_line_style(ShapeStyle{
                    color: WHITE.to_rgba(),
                    filled: false,
                    stroke_width: 0
                });
            },
            true => {}
        }

        match $self.plots.plotting_settings.show_x_axis {
            false => {
                $axis.disable_x_axis();
            },
            _ => {}
        }

        match $self.plots.plotting_settings.show_y_axis {
            false => {
                $axis.disable_y_axis();
            },
            _ => {}
        }

        match $self.plots.plotting_settings.show_x_mesh {
            false => {
                $axis.disable_x_mesh();
            },
            _ => {}
        }

        match $self.plots.plotting_settings.show_y_mesh {
            false => {
                $axis.disable_y_mesh();
            },
            _ => {}
        }

        $axis
        .x_desc($self.plots.x_label.clone())
        .y_desc($self.plots.y_label.clone())
        .x_label_style({
            TextStyle::from(("sans-serif", $self.plots.plotting_settings.label_font_size).into_font()).pos(Pos::new(HPos::Right, VPos::Top))
        })
        .y_label_style({
            TextStyle::from(("sans-serif", $self.plots.plotting_settings.label_font_size).into_font()).pos(Pos::new(HPos::Right, VPos::Center))
        })
        .set_tick_mark_size(LabelAreaPosition::Bottom, $self.plots.plotting_settings.x_tick_mark_size)
        .set_tick_mark_size(LabelAreaPosition::Left, $self.plots.plotting_settings.y_tick_mark_size)
        .x_label_offset($self.plots.plotting_settings.x_label_offset)
        .y_label_offset($self.plots.plotting_settings.y_label_offset)
        .draw().unwrap();
    };
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
                    self.simple_2d_plot(root, true)
                },
                SupportedPlotTypes::Lines3d => {
                    self.simple_3d_plot(root, true)
                },
                SupportedPlotTypes::HeatMap => {
                    self.heatmap_plot(root, true)
                },
                SupportedPlotTypes::HeatMapAndLines2d => {
                    self.heatmap_plot(root, true);
                    self.simple_2d_plot(root, false)
                },
                SupportedPlotTypes::Contour => {
                    self.contour_plot(root, true)
                },
                SupportedPlotTypes::Polygon => {
                    self.polygon_plot(root, true)
                },
                SupportedPlotTypes::ContourAndLines2d => {
                    self.contour_plot(root, true);
                    self.simple_2d_plot(root, false)
                },
                SupportedPlotTypes::HeatMapContourAndLines2d => {
                    self.heatmap_plot(root, true);
                    self.contour_plot(root, true);
                    self.simple_2d_plot(root, false)
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

        fn simple_2d_plot(&self, root: &mut DrawingArea<OutputType, Shift>, is_primary_plot: bool) {

            let range_x_f64 = self.plots.x_range.as_ref().unwrap().start as f64 .. self.plots.x_range.as_ref().unwrap().end as f64;
            let range_y_f64 = self.plots.y_range.as_ref().unwrap().start as f64 .. self.plots.y_range.as_ref().unwrap().end as f64;
            let range_y2_f64 = self.plots.y2_range.as_ref().unwrap().start as f64 .. self.plots.y2_range.as_ref().unwrap().end as f64;
            // let range_y2_f64_dummy = 1. as f64 .. 1.1 as f64;

            // using unwrap 
            let x_axis_scaling = self.plots.x_axis_scaling.clone().expect("x axis scaling should have been set!");
            let y_axis_scaling = self.plots.y_axis_scaling.clone().expect("y axis scaling should have been set!");
            let y2_axis_scaling = self.plots.y2_axis_scaling.clone().expect("y2 axis scaling should have been set!");

            let mut chart = ChartBuilder::on(&root);
            let mut chart = make_std_chart!(self, chart);
   
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

        fn simple_3d_plot(&self, root: &mut DrawingArea<OutputType, Shift>, is_primary_plot: bool) {

            let range_x_f64 = self.plots.x_range.as_ref().unwrap().start as f64 .. self.plots.x_range.as_ref().unwrap().end as f64;
            let range_y_f64 = self.plots.y_range.as_ref().unwrap().start as f64 .. self.plots.y_range.as_ref().unwrap().end as f64;
            let range_z_f64 = self.plots.z_range.as_ref().unwrap().start as f64 .. self.plots.z_range.as_ref().unwrap().end as f64;

            // using unwrap 
            let x_axis_scaling = self.plots.x_axis_scaling.clone().expect("x axis scaling should have been set!");
            let y_axis_scaling = self.plots.y_axis_scaling.clone().expect("y axis scaling should have been set!");
            let z_axis_scaling = self.plots.z_axis_scaling.clone().expect("z axis scaling should have been set!");

            let mut chart = ChartBuilder::on(&root);
            let mut chart = make_std_chart!(self, chart);

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

        fn heatmap_plot(&self, root: &mut DrawingArea<OutputType, Shift>, is_primary_plot: bool) {

            let range_x_f64 = self.plots.x_range.as_ref().unwrap().start as f64 .. self.plots.x_range.as_ref().unwrap().end as f64;
            let range_y_f64 = self.plots.y_range.as_ref().unwrap().start as f64 .. self.plots.y_range.as_ref().unwrap().end as f64;
            let range_y2_f64 = self.plots.y2_range.as_ref().unwrap().start as f64 .. self.plots.y2_range.as_ref().unwrap().end as f64;
            let range_z_f64 = self.plots.z_range.as_ref().unwrap().start as f64 .. self.plots.z_range.as_ref().unwrap().end as f64;

            // using unwrap 
            let x_axis_scaling = self.plots.x_axis_scaling.clone().expect("x axis scaling should have been set!");
            let y_axis_scaling = self.plots.y_axis_scaling.clone().expect("y axis scaling should have been set!");
            let z_axis_scaling = self.plots.z_axis_scaling.clone().expect("z axis scaling should have been set!");

            let mut chart = ChartBuilder::on(&root);
            let mut chart = make_std_chart!(self, chart);

            match (x_axis_scaling, y_axis_scaling) {
                (PlotAxisScaling::Linear, PlotAxisScaling::Linear) => {
                    let mut chart = chart
                        .build_cartesian_2d(range_x_f64.clone(), range_y_f64)
                        .unwrap()
                        .set_secondary_coord((range_x_f64), (range_y2_f64));

                    let mut axis = chart.configure_mesh(); // all of this is stolen from "produce_other_2d_plot_settings" maybe be smarter about this!

            
                    make_std_axis!(axis, self);

                    match self.plots.y2_axis_scaling.as_ref().unwrap() {
                        PlotAxisScaling::NoAxis => {},
                        _ => {
                            // process the settings for the secondary axis, maybe this can be made nicer
                            let secondary_axis_label = self.plots.y2_label.clone();
            
                            chart
                                .configure_secondary_axes()
                                .y_desc(secondary_axis_label)
                                .draw().unwrap();
                        }
                    }

                    macro_rules! draw_the_heatmap {
                        ($plot: expr, $chart: expr, $x_values: expr, $y_values: expr, $z_values: expr) => {
                            let n_points_x = $x_values.len();
                            let n_points_y = $y_values.len();
        
                            let mut lowest_x = $x_values[0] as f64;
                            let mut highest_x = $x_values[0] as f64;
                            let x_values_converted = $x_values.iter().map(|x| {
                                if (*x as f64) < lowest_x {lowest_x = *x as f64}; 
                                if (*x as f64) > highest_x {highest_x = *x as f64};
                                *x as f64
                            }).collect::<Vec<f64>>();
        
                            let dx_per_node = (highest_x - lowest_x) as f64 / (n_points_x as f64);
        
                            let mut lowest_y = $y_values[0] as f64;
                            let mut highest_y = $y_values[0] as f64;
                            let y_values_converted = $y_values.iter().map(|y| {
                                if (*y as f64) < lowest_y {lowest_y = *y as f64}; 
                                if (*y as f64 )> highest_y {highest_y = *y as f64};
                                *y as f64
                            }).collect::<Vec<f64>>();
        
                            let dy_per_node = (highest_y - lowest_y) / (n_points_y as f64);
                            
                            let mut lowest_z = $z_values[0][0] as f64;
                            let mut highest_z = $z_values[0][0] as f64;
                            let z_values_for_iter = $z_values.clone();
                            let _ = z_values_for_iter.iter().map(|z_row| {
                                let mut row_lowest = z_row[0] as f64;
                                let mut row_highest = z_row[0] as f64;
                                let _ = z_row.iter().map( |z|{
                                    if (*z as f64) < row_lowest {row_lowest = *z as f64}; 
                                    if (*z as f64) > row_highest {row_highest = *z as f64};
                                }).collect::<Vec<()>>();
                                if (row_lowest as f64) < lowest_z {lowest_z = row_lowest as f64}; 
                                if (row_highest as f64) > highest_z {highest_z = row_highest as f64};
                            }).collect::<Vec<()>>();

                            // let color_map = $plot.surface_3d.clone().unwrap().color_map;

                            chart.draw_series(
                                y_values_converted.iter().enumerate().flat_map( |(y_index, _)|{
                                    let z_row = (z_values_for_iter[y_index]).clone();
                                    let colormap = $plot.surface_3d.clone().unwrap().color_map.clone();
                                    x_values_converted.iter().enumerate().map( move |(x_index, _)|{
                                        Rectangle::new(
                                            [(dx_per_node*(x_index as f64)+lowest_x, dy_per_node*(y_index as f64)+lowest_y),(dx_per_node*(x_index as f64 + 1.)+lowest_x, dy_per_node*(y_index as f64 +1.)+lowest_y)],
                                            ShapeStyle {
                                                color: ColorMaps::get_color_from_map(vec![(((((z_row[x_index]).clone() as f32 - lowest_z as f32) / ((highest_z - lowest_z) as f32))).powf(1./3.))], 
                                                colormap.clone()),
                                                filled: true,
                                                stroke_width: 0
                                            }
                                        )
                                    })
                                })
                            ).unwrap();
                        };
                    }

                    match (self.plots.surface_3d.as_ref().unwrap().x_values.clone(), self.plots.surface_3d.as_ref().unwrap().y_values.clone(), self.plots.surface_3d.as_ref().unwrap().z_values.clone(), self.plots.surface_3d.as_ref().unwrap().z_function.clone()) {
                        (Option::Some(x_values), Option::Some(y_values),Option::Some(z_values), _) => {
                            draw_the_heatmap!(self.plots, chart, x_values, y_values, z_values);
                        },
                        (_,_,_,Option::Some(z_function)) => {
                            let x_values = f64::linspace(self.plots.x_range.clone().unwrap().start as f64, self.plots.x_range.clone().unwrap().end as f64, self.plots.plotting_settings.heatmap_n_points);
                            let y_values = f64::linspace(self.plots.y_range.clone().unwrap().start as f64, self.plots.y_range.clone().unwrap().end as f64, self.plots.plotting_settings.heatmap_n_points);
                            let z_values: Vec<Vec<f64>> = y_values.iter().map(|y| {
                                x_values.iter().map(|x| {
                                    (z_function)(vec![*x,*y])
                                }).collect()
                            }).collect();
                            draw_the_heatmap!(self.plots, chart, x_values, y_values, z_values);
                        },
                        _ => {}
                    }



                    // todo!(); // this one took 10 minutes to find
                },
                _ => {
                    panic!("Only linear plot axis are supported in heat maps!")
                },
            }
        }

        fn contour_plot(&self, root: &mut DrawingArea<OutputType, Shift>, is_primary_plot: bool) {

            let range_x_f64 = self.plots.x_range.as_ref().unwrap().start as f64 .. self.plots.x_range.as_ref().unwrap().end as f64;
            let range_y_f64 = self.plots.y_range.as_ref().unwrap().start as f64 .. self.plots.y_range.as_ref().unwrap().end as f64;
            let range_y2_f64 = self.plots.y2_range.as_ref().unwrap().start as f64 .. self.plots.y2_range.as_ref().unwrap().end as f64;
            let range_z_f64 = self.plots.z_range.as_ref().unwrap().start as f64 .. self.plots.z_range.as_ref().unwrap().end as f64;

            // using unwrap 
            let x_axis_scaling = self.plots.x_axis_scaling.clone().expect("x axis scaling should have been set!");
            let y_axis_scaling = self.plots.y_axis_scaling.clone().expect("y axis scaling should have been set!");
            let z_axis_scaling = self.plots.z_axis_scaling.clone().expect("z axis scaling should have been set!");

            let mut chart = ChartBuilder::on(&root);
            let mut chart = make_std_chart!(self, chart);

            match (x_axis_scaling, y_axis_scaling) {
                (PlotAxisScaling::Linear, PlotAxisScaling::Linear) => {
                    let mut chart = chart
                        .build_cartesian_2d(range_x_f64.clone(), range_y_f64)
                        .unwrap()
                        .set_secondary_coord((range_x_f64).log_scale(), (range_y2_f64));

                    let mut axis = chart.configure_mesh(); // all of this is stolen from "produce_other_2d_plot_settings" maybe be smarter about this!

                    make_std_axis!(axis, self);

                    match self.plots.y2_axis_scaling.as_ref().unwrap() {
                        PlotAxisScaling::NoAxis => {},
                        _ => {
                            // process the settings for the secondary axis, maybe this can be made nicer
                            let secondary_axis_label = self.plots.y2_label.clone();
            
                            chart
                                .configure_secondary_axes()
                                .y_desc(secondary_axis_label)
                                .draw().unwrap();
                        }
                    }

                    macro_rules! draw_the_contour_plot {
                        ($plot: expr, $chart: expr, $x_values: expr, $y_values: expr, $z_values: expr) => {
                            let n_points_x = $x_values.len();
                            let n_points_y = $y_values.len();
        
                            let mut lowest_x = $x_values[0] as f64;
                            let mut highest_x = $x_values[0] as f64;
                            let x_values_converted = $x_values.iter().map(|x| {
                                if (*x as f64) < lowest_x {lowest_x = *x as f64}; 
                                if (*x as f64) > highest_x {highest_x = *x as f64};
                                *x as f64
                            }).collect::<Vec<f64>>();
        
                            let dx_per_node = (highest_x - lowest_x) as f64 / (n_points_x as f64);
        
                            let mut lowest_y = $y_values[0] as f64;
                            let mut highest_y = $y_values[0] as f64;
                            let y_values_converted = $y_values.iter().map(|y| {
                                if (*y as f64) < lowest_y {lowest_y = *y as f64}; 
                                if (*y as f64 )> highest_y {highest_y = *y as f64};
                                *y as f64
                            }).collect::<Vec<f64>>();
        
                            let dy_per_node = (highest_y - lowest_y) / (n_points_y as f64);
                            
                            let mut lowest_z = $z_values[0][0] as f64;
                            let mut highest_z = $z_values[0][0] as f64;
                            let z_values_for_iter = $z_values.clone();
                            let _ = z_values_for_iter.iter().map(|z_row| {
                                let mut row_lowest = z_row[0] as f64;
                                let mut row_highest = z_row[0] as f64;
                                let _ = z_row.iter().map( |z|{
                                    if (*z as f64) < row_lowest {row_lowest = *z as f64}; 
                                    if (*z as f64) > row_highest {row_highest = *z as f64};
                                }).collect::<Vec<()>>();
                                if (row_lowest as f64) < lowest_z {lowest_z = row_lowest as f64}; 
                                if (row_highest as f64) > highest_z {highest_z = row_highest as f64};
                            }).collect::<Vec<()>>();

                            let division_factor =  1. / $plot.plotting_settings.contour_n_lines as f64;
                            let band_size = $plot.plotting_settings.contour_band_width as f64 /$plot.plotting_settings.contour_n_lines as f64;

                            let contour_alpha_factor: f64;
                            if let Option::Some(factor) = $plot.contour.clone().unwrap().contour_alpha_factor {
                                contour_alpha_factor = factor;
                            } else {
                                contour_alpha_factor = $plot.plotting_settings.contour_alpha_value;
                            }

                            chart.draw_series(
                                y_values_converted
                                    .iter()
                                    .enumerate()
                                    .flat_map( |(y_index, _)|{
                                        let z_row = (z_values_for_iter[y_index]).clone();
                                        let z_row_colorfilter = (z_values_for_iter[y_index]).clone();
                                        let colormap = $plot.contour.clone().unwrap().color_map.clone();


                                        x_values_converted
                                            .iter()
                                            .enumerate()
                                            .filter(move |(x_index, _)| {
                                                let normalized_value: f64 = (z_row_colorfilter[*x_index].clone() as f64 - lowest_z  ) / ((highest_z - lowest_z) );
                                                // if normalized_value % (division_factor as f64) < band_size && normalized_value > 1. / division_factor {
                                                if (normalized_value)% (division_factor as f64) < band_size && normalized_value > division_factor{
                                                    true
                                                } else {
                                                    false
                                                }
                                            })
                                            .map( move |(x_index, _)|{
                                                Rectangle::new(
                                                    [(dx_per_node*(x_index as f64)+lowest_x, dy_per_node*(y_index as f64)+lowest_y),(dx_per_node*(x_index as f64 + 1.)+lowest_x, dy_per_node*(y_index as f64 +1.)+lowest_y)],
                                                    ShapeStyle {
                                                        color: {
                                                            let mut color = ColorMaps::get_color_from_map(vec![(((((z_row[x_index]).clone() as f32 - lowest_z as f32) / ((highest_z - lowest_z) as f32))))], colormap.clone());
                                                            color.3 = color.3 * contour_alpha_factor;
                                                            color
                                                        },
                                                        filled: true,
                                                        stroke_width: 0
                                                    }
                                                )
                                            })
                                    })
                            ).unwrap();
                        };
                    }

                    match (self.plots.contour.as_ref().unwrap().x_values.clone(), self.plots.contour.as_ref().unwrap().y_values.clone(), self.plots.contour.as_ref().unwrap().z_values.clone(), self.plots.contour.as_ref().unwrap().z_function.clone()) {
                        (Option::Some(x_values), Option::Some(y_values),Option::Some(z_values), _) => {
                            draw_the_contour_plot!(self.plots, chart, x_values, y_values, z_values);
                        },
                        (_,_,_,Option::Some(z_function)) => {
                            let x_values = f64::linspace(self.plots.x_range.clone().unwrap().start as f64, self.plots.x_range.clone().unwrap().end as f64, self.plots.plotting_settings.contour_n_points);
                            let y_values = f64::linspace(self.plots.y_range.clone().unwrap().start as f64, self.plots.y_range.clone().unwrap().end as f64, self.plots.plotting_settings.contour_n_points);
                            let z_values: Vec<Vec<f64>> = y_values.iter().map(|y| {
                                x_values.iter().map(|x| {
                                    (z_function)(vec![*x,*y])
                                }).collect()
                            }).collect();
                            draw_the_contour_plot!(self.plots, chart, x_values, y_values, z_values);
                        },
                        _ => {}
                    }



                    // todo!(); // this one took 10 minutes to find
                },
                _ => {
                    panic!("Only linear plot axis are supported in heat maps!")
                },
            }
        }

        fn polygon_plot(&self, root: &mut DrawingArea<OutputType, Shift>, is_primary_plot: bool) {

            let range_x_f64 = self.plots.x_range.as_ref().unwrap().start as f64 .. self.plots.x_range.as_ref().unwrap().end as f64;
            let range_y_f64 = self.plots.y_range.as_ref().unwrap().start as f64 .. self.plots.y_range.as_ref().unwrap().end as f64;
            let range_y2_f64 = self.plots.y2_range.as_ref().unwrap().start as f64 .. self.plots.y2_range.as_ref().unwrap().end as f64;
            // let range_y2_f64_dummy = 1. as f64 .. 1.1 as f64;

            // using unwrap 
            let x_axis_scaling = self.plots.x_axis_scaling.clone().expect("x axis scaling should have been set!");
            let y_axis_scaling = self.plots.y_axis_scaling.clone().expect("y axis scaling should have been set!");
            let y2_axis_scaling = self.plots.y2_axis_scaling.clone().expect("y2 axis scaling should have been set!");
            
            match (x_axis_scaling, y_axis_scaling) {
                (PlotAxisScaling::Linear, PlotAxisScaling::Linear) => {},
                _ => {panic!("Let's burn that bridge of non linear polygon plots once we get there")}
            }
            

            let mut chart = ChartBuilder::on(&root);
            let mut chart = make_std_chart!(self, chart);

                let mut chart = chart
                .build_cartesian_2d(range_x_f64.clone(), range_y_f64)
                .unwrap()
                .set_secondary_coord((range_x_f64), (range_y2_f64));

            let mut axis = chart.configure_mesh(); // all of this is stolen from "produce_other_2d_plot_settings" maybe be smarter about this!

            make_std_axis!(axis, self);

            match self.plots.y2_axis_scaling.as_ref().unwrap() {
                PlotAxisScaling::NoAxis => {},
                _ => {
                    // process the settings for the secondary axis, maybe this can be made nicer
                    let secondary_axis_label = self.plots.y2_label.clone();
    
                    chart
                        .configure_secondary_axes()
                        .y_desc(secondary_axis_label)
                        .draw().unwrap();
                }
            }

            let lowest_z_in_plot = self.plots.z_range.as_ref().unwrap().start;
            let highest_z_in_plot = self.plots.z_range.as_ref().unwrap().end;
            let z_range_plot: f32;
            if (lowest_z_in_plot as f32) - (highest_z_in_plot as f32) == 0. {
                z_range_plot = 1.
            } else {
                z_range_plot = highest_z_in_plot as f32 - lowest_z_in_plot as f32;
            }
            
            let number_of_polygon = self.plots.polygons.as_ref().unwrap().polygons.len();

            for (number, polygon) in self.plots.polygons.as_ref().unwrap().polygons.iter().enumerate() {
                let z_value = polygon.z_value as f32;
                let polygon_color: RGBAColor;
                match (polygon.color, self.plots.polygons.as_ref().unwrap().color) {
                    (Option::Some(color), _) => {
                        // priotity: individual polygon color
                        polygon_color = color;
                    },
                    (Option::None, Option::Some(color)) => {
                        // 2nd priority: constant polygon set coloring
                        polygon_color = color;
                    },
                    _ => {
                        // reverting back to polygon set colormap
                        let mut index_of_line: f32;
        
                        
                        match self.plots.polygons.as_ref().unwrap().color_map {
                            PlotBuilderColorMaps::Palette99 => {
                                index_of_line = (number / number_of_polygon) as f32;
                            },
                            PlotBuilderColorMaps::Palette99ReversedOrder => {
                                index_of_line = (number / number_of_polygon) as f32;
                            },
                            _ => {
                                index_of_line = ((z_value - lowest_z_in_plot as f32) / z_range_plot) as f32;
                                index_of_line = (self.plots.plotting_settings.color_map_restricter_upper_bound - self.plots.plotting_settings.color_map_restricter_lower_bound) * index_of_line 
                                                + self.plots.plotting_settings.color_map_restricter_lower_bound;
                            }
                        }
                        polygon_color = ColorMaps::get_color_from_map(vec![index_of_line, number_of_polygon as f32], self.plots.polygons.as_ref().unwrap().color_map.clone())
                    }
                }

                let line_width: u32;
                match (polygon.line_width, self.plots.polygons.as_ref().unwrap().line_width) {
                    (Option::Some(line_width_polygon),_) => {
                        line_width = line_width_polygon;
                    },
                    (Option::None, Option::Some(line_width_polygon)) => {
                        line_width = line_width_polygon;
                    }
                    _ => {
                        line_width = self.plots.plotting_settings.polygon_line_width;
                    }
                }

                let polygon_is_filled: bool;
                match (polygon.filled, self.plots.polygons.as_ref().unwrap().filled) {
                    (Option::Some(filled_bool),_) => {
                        polygon_is_filled = filled_bool;
                    },
                    (Option::None, Option::Some(filled_bool)) => {
                        polygon_is_filled = filled_bool;
                    }
                    _ => {
                        polygon_is_filled = self.plots.plotting_settings.polygon_filled;
                    }
                }

                let style = ShapeStyle {color: polygon_color.clone(), filled: polygon_is_filled, stroke_width: line_width};

                let polygon_x_values = polygon.x_values.clone();
                let polygon_y_values = polygon.y_values.clone();

                let (backend_base_x, backend_base_y) = chart.backend_coord(&(polygon_x_values[0] as f64, polygon_y_values[0] as f64));

                let collected_points = polygon.x_values
                    .iter()
                    .zip(&polygon.y_values)
                    .map(|(x,y)| {
                        let base_x = backend_base_x.clone();
                        let base_y = backend_base_y.clone();
                        let (coordinates_backend_x , coordinates_backend_y) = chart.backend_coord(&(*x as f64, *y as f64));
                        (coordinates_backend_x-base_x, coordinates_backend_y-base_y)
                    }).collect::<Vec<(i32,i32)>>();

                // let polygon_converted_to_plotters = plotters::element::Polygon::new(
                //     collected_points,
                //     style
                // );

                // let coordinates = chart.backend_coord();

                chart   
                    .draw_series(
                        (0..1).map(|_|{
                            EmptyElement::at((polygon_x_values[0] as f64, polygon_y_values[0] as f64))
                            + plotters::element::Polygon::new(&*collected_points, style)
                        })
                    ) 
                    .unwrap();
                
            }


        }
    }
}}

macro_rules! impl_plot_processor_plotters_backend_functions_per_type {
    ($T: ident) => {
        impl PlotProcessorPlottersBackendFunctions<$T> for PlotProcessor<$T, PlottersBackend> {
        
            fn SVG_to_file(&mut self, file_path: &str) {
                self.plots.plotting_settings.x_tick_mark_size = (self.plots.plotting_settings.x_tick_mark_size as f64 * 1.1) as i32;
                self.plots.plotting_settings.y_tick_mark_size = (self.plots.plotting_settings.y_tick_mark_size as f64 * 1.5) as i32;
                let (width, height) = self.plots.get_plot_dimensions();

                let mut root = SVGBackend::new(file_path, (width, height)).into_drawing_area();

                self.process(&mut root);
            }

            fn SVG_to_mem(&mut self) -> String {
                self.plots.plotting_settings.x_tick_mark_size = (self.plots.plotting_settings.x_tick_mark_size as f64 * 1.1) as i32;
                let (width, height) = self.plots.get_plot_dimensions(); 

                let mut string_buffer = String::new();
                {
                    let mut root = SVGBackend::with_string(&mut string_buffer, (width, height)).into_drawing_area();
                    self.process(&mut root);
                }
                string_buffer
            }
            
            fn SVG_to_RGBA(&mut self) -> Vec<u8> {
                self.plots.plotting_settings.x_tick_mark_size = (self.plots.plotting_settings.x_tick_mark_size as f64 * 1.1) as i32;
                // use plotters svg backend to render a svg image
                // then use usvg to render svg to rgba
                let start_svg = Instant::now();
                let svg_string = self.SVG_to_mem();
                let time_svg = start_svg.elapsed();
                let parsing_svg = Instant::now();
                let parsing_options = resvg::usvg::Options::default();
                let parsed_svg = &usvg::TreeParsing::from_str(&svg_string, &parsing_options).unwrap();
                let resvg_tree = resvg::Tree::from_usvg(parsed_svg);
                let time_parsing = parsing_svg.elapsed();
                let start_render_2 = Instant::now();
                let mut pixmap = Pixmap::new(self.plots.plot_width.unwrap(), self.plots.plot_height.unwrap()).expect("Error creating pixmap");
                resvg_tree.render(Default::default(), &mut pixmap.as_mut());
                let time_render_2 = start_render_2.elapsed();
            
                pixmap.take()
            }

            fn bitmap_to_rgb(&mut self) -> Vec<u8> {
                self.plots.plotting_settings.x_tick_mark_size = (self.plots.plotting_settings.x_tick_mark_size as f64 * 0.5) as i32;
                self.plots.plotting_settings.y_label_offset = (self.plots.plotting_settings.y_label_offset as f64 * 0.) as i32;
                let (width, height) = self.plots.get_plot_dimensions(); 

                let mut buffer = vec![0u8; (width*height*3) as usize];
                {
                    let mut root = BitMapBackend::with_buffer(&mut buffer, (width, height)).into_drawing_area();
                    self.process(&mut root);
                }
                

                buffer
            }

            fn bitmap_to_file(&mut self, file_path: &str) {
                self.plots.plotting_settings.x_tick_mark_size = (self.plots.plotting_settings.x_tick_mark_size as f64 * 0.5) as i32;
                self.plots.plotting_settings.y_label_offset = (self.plots.plotting_settings.y_label_offset as f64 *0.) as i32;

                self.plots.plotting_settings.plotters_legend_bar_shift_y = (self.plots.plotting_settings.plotters_legend_bar_shift_y as f64 *0.0) as i32;
                let (width, height) = self.plots.get_plot_dimensions();  
                {
                    let mut root = BitMapBackend::new(file_path, (width, height)).into_drawing_area();
                    self.process(&mut root);
                }
            }
            
        } 
    }
}

macro_rules! produce_other_2d_plot_settings {
    ($chart: expr, $self: expr) => {
        let mut axis = $chart.configure_mesh();

        // match $self.plots.plotting_settings.x_grid_major_subdevisions {
        //     Option::Some(n_lines) => {
        //         axis.x_labels(n_lines);
        //     },
        //     _ => {} // default is to already show major gridlines
        // }

        // match $self.plots.plotting_settings.y_grid_major_subdevisions {
        //     Option::Some(n_lines) => {
        //         axis.y_labels(n_lines);
        //     },
        //     _ => {} // default is to already show major gridlines
        // }

        // match $self.plots.plotting_settings.show_x_grid_minor {
        //     true => {
        //         match $self.plots.plotting_settings.x_grid_minor_subdevisions {
        //             Option::Some(n_lines) => {
        //                 axis.x_max_light_lines(n_lines);
        //             },
        //             _ => {} // default is to already show minor gridlines
        //         }
        //     },
        //     false => {
        //         axis.x_max_light_lines(0);
        //     }
        // }

        // match $self.plots.plotting_settings.show_y_grid_minor {
        //     true => {
        //         match $self.plots.plotting_settings.y_grid_minor_subdevisions {
        //             Option::Some(n_lines) => {
        //                 axis.y_max_light_lines(n_lines);
        //             },
        //             _ => {} // default is to already show minor gridlines
        //         }
        //     },
        //     false => {
        //         axis.y_max_light_lines(0);
        //     }
        // }

        // match $self.plots.plotting_settings.show_grid_major {
        //     false => {
        //         axis.bold_line_style(ShapeStyle{
        //             color: WHITE.to_rgba(),
        //             filled: false,
        //             stroke_width: 0
        //         });
        //     },
        //     true => {}
        // }

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

        make_std_axis!(axis, $self);
        
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
            } else if let Option::Some(color_map) = &series.color_map {
                let mut index_of_line: f32;

                index_of_line = number as f32 / amount_of_lines as f32;
                match color_map {
                    PlotBuilderColorMaps::Palette99 => { },
                    PlotBuilderColorMaps::Palette99ReversedOrder => {},
                    _ => {
                        index_of_line = ($self.plots.plotting_settings.color_map_restricter_upper_bound - $self.plots.plotting_settings.color_map_restricter_lower_bound) * index_of_line 
                                        + $self.plots.plotting_settings.color_map_restricter_lower_bound;
                    }
                }

                line_color = ColorMaps::get_color_from_map(vec![index_of_line, amount_of_lines as f32], color_map.clone())
            } else {
                if let Option::Some(color) = $self.plots.plotting_settings.line_color {
                    line_color = color;
                } else {
                    let mut index_of_line: f32;

                    index_of_line = number as f32 / amount_of_lines as f32;

                    match $self.plots.plotting_settings.color_map_line.clone() {
                        PlotBuilderColorMaps::Palette99 => { },
                        PlotBuilderColorMaps::Palette99ReversedOrder => {},
                        _ => {
                            index_of_line = ($self.plots.plotting_settings.color_map_restricter_upper_bound - $self.plots.plotting_settings.color_map_restricter_lower_bound) * index_of_line 
                                            + $self.plots.plotting_settings.color_map_restricter_lower_bound;
                        }
                    }

                    line_color = ColorMaps::get_color_from_map(vec![index_of_line, amount_of_lines as f32], $self.plots.plotting_settings.color_map_line.clone());
                }
            }
            let style = ShapeStyle {color: line_color.clone(), filled: marker_fill, stroke_width: line_width};
            let legend_style = ShapeStyle {color: line_color.clone(), filled: true, stroke_width: 1};
            let bar_size = $self.plots.plotting_settings.plotters_legend_bar_size;
            let bar_shift_x = $self.plots.plotting_settings.plotters_legend_bar_shift_x;
            let bar_shift_y = $self.plots.plotting_settings.plotters_legend_bar_shift_y;
            if !(series.on_secondary_axis) {
                $chart
                    .draw_series(LineSeries::new(series.x_values.iter().zip(series.y_values.clone()).map(|(x, y)| (*x as f64, y as f64)), style)
                    .point_size(marker_size))
                    .unwrap()
                    .label(legend_name)
                    .legend(move |(x,y)| Rectangle::new([(x - bar_shift_x, y - bar_shift_y), (x, y-bar_shift_y-bar_size)], legend_style));
            } else {
                $chart
                    .draw_secondary_series(LineSeries::new(series.x_values.iter().zip(series.y_values.clone()).map(|(x, y)| (*x as f64, y as f64)), style)
                    .point_size(marker_size))
                    .unwrap()
                    .label(legend_name)
                    .legend(move |(x,y)| Rectangle::new([(x - bar_shift_x, y ), (x, y - bar_shift_y -bar_size)], legend_style));
            }            
        }

        let legend_location: SeriesLabelPosition;
        match $self.plots.plotting_settings.legend_location {
            LegendPosition::NorthEast => {legend_location = SeriesLabelPosition::UpperRight},
            LegendPosition::North => {legend_location = SeriesLabelPosition::UpperMiddle},
            LegendPosition::NorthWest => {legend_location = SeriesLabelPosition::UpperLeft},
            LegendPosition::West => {legend_location = SeriesLabelPosition::MiddleLeft},
            LegendPosition::SouthWest => {legend_location = SeriesLabelPosition::LowerLeft},
            LegendPosition::South => {legend_location = SeriesLabelPosition::LowerMiddle},
            LegendPosition::SouthEast => {legend_location = SeriesLabelPosition::LowerRight},
            LegendPosition::East => {legend_location = SeriesLabelPosition::MiddleRight}
        }

        if $self.plots.plotting_settings.legend_show {
            $chart
            .configure_series_labels()
            .position(legend_location)
            .margin($self.plots.plotting_settings.plotters_legend_margin)
            .legend_area_size($self.plots.plotting_settings.plotters_legend_area_size)
            .border_style(GREY)
            .background_style(GREY.mix($self.plots.plotting_settings.plotters_legend_transparancy))
            .label_font(("Calibri", $self.plots.plotting_settings.plotters_legend_font_size))
            .draw().unwrap();
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

        match $self.plots.plotting_settings.show_grid_major {
            false => {
                // probably not even possible to plot grid major in 3d?
                // it is! but in different function! look at configure_axes()
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
            } else if let Option::Some(color_map) = &series.color_map {
                let index_of_line: f32;

                index_of_line = number as f32 / amount_of_lines as f32;
                line_color = ColorMaps::get_color_from_map(vec![index_of_line, amount_of_lines as f32], color_map.clone())
            } else {
                if let Option::Some(color) = $self.plots.plotting_settings.line_color {
                    line_color = color;
                } else {
                    let index_of_line: f32;

                    index_of_line = number as f32 / amount_of_lines as f32;

                    line_color = ColorMaps::get_color_from_map(vec![index_of_line, amount_of_lines as f32], $self.plots.plotting_settings.color_map_line.clone());
                }
            }

            let style = ShapeStyle {color: line_color, filled: marker_fill, stroke_width: line_width};
            let legend_style = ShapeStyle {color: line_color.clone(), filled: true, stroke_width: 1};
            let bar_size = $self.plots.plotting_settings.plotters_legend_bar_size;
            let bar_shift_x = $self.plots.plotting_settings.plotters_legend_bar_shift_x;
            let bar_shift_y = $self.plots.plotting_settings.plotters_legend_bar_shift_y;
            $chart
            // .draw_series(LineSeries::new(series.x_values.iter().zip(series.y_values.clone()).map(|(x, y)| (*x as f64, y as f64)), style)
            .draw_series(LineSeries::new(
                (0..series.x_values.len()).map(|index|
                (series.x_values[index] as f64, series.y_values[index] as f64, series.z_values[index] as f64)),
            style)
            .point_size(marker_size))
            .unwrap()
            .label(legend_name)
            .legend(move |(x,y)| Rectangle::new([(x - bar_shift_x, y ), (x, y - bar_shift_y -bar_size)], legend_style));
            

            let legend_location: SeriesLabelPosition;
            match $self.plots.plotting_settings.legend_location {
                LegendPosition::NorthEast => {legend_location = SeriesLabelPosition::UpperRight},
                LegendPosition::North => {legend_location = SeriesLabelPosition::UpperMiddle},
                LegendPosition::NorthWest => {legend_location = SeriesLabelPosition::UpperLeft},
                LegendPosition::West => {legend_location = SeriesLabelPosition::MiddleLeft},
                LegendPosition::SouthWest => {legend_location = SeriesLabelPosition::LowerLeft},
                LegendPosition::South => {legend_location = SeriesLabelPosition::LowerMiddle},
                LegendPosition::SouthEast => {legend_location = SeriesLabelPosition::LowerRight},
                LegendPosition::East => {legend_location = SeriesLabelPosition::MiddleRight}
            }
    
            if $self.plots.plotting_settings.legend_show {
                $chart
                .configure_series_labels()
                .position(legend_location)
                .margin($self.plots.plotting_settings.plotters_legend_margin)
                .legend_area_size($self.plots.plotting_settings.plotters_legend_area_size)
                .border_style(GREY)
                .background_style(GREY.mix($self.plots.plotting_settings.plotters_legend_transparancy))
                .label_font(("Calibri", $self.plots.plotting_settings.plotters_legend_font_size))
                .draw().unwrap();
            }
        }
    }
}



impl_plot_processor_plotters_backend_functions_per_type!(f32);
impl_plot_processor_plotters_backend_functions_per_type!(f64);



impl_plot_processor_plotters_backend_functions_per_type_with_annoying_variants!(f32);
impl_plot_processor_plotters_backend_functions_per_type_with_annoying_variants!(f64);
