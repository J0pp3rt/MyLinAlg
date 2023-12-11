pub mod simulation_constants;
pub use simulation_constants::{*};

pub mod line_search_methods;
pub use line_search_methods::{*};

pub mod decent_methods;
pub use decent_methods::{*};

use mla::{*, PolynomialFunctions};
use plotters::prelude::RGBAColor;
use std::ops::Not;

use std::marker::PhantomData;
use std::rc::Rc;

use std::io;

fn main() {

    let optimization_settings = OptimizationSettings {
        initial_point: PosNDof { x: vec![-2., 3.] },
        extrapolation_method: LineSearchExtrapolationMethods::GoldenRuleExtrapolation,
        interpolation_method: LineSearchInterpolationMethods::GoldenRuleInterpolation,
        // descent_method: DescentMethodsEnum::SteepestDescent,
        // descent_method: DescentMethodsEnum::ConjugateDescent(ConjugateGradientMethods::FletcherReeves),
        // descent_method: DescentMethodsEnum::ConjugateDescent(ConjugateGradientMethods::PolakRibiere),
        // descent_method: DescentMethodsEnum::QuasiNewton(QuasiNewtonMethods::BFGS),
        // descent_method: DescentMethodsEnum::QuasiNewton(QuasiNewtonMethods::BFGSBeforeInverse),
        // descent_method: DescentMethodsEnum::QuasiNewton(QuasiNewtonMethods::BFGSBeforeInverseMoreStable),
        // descent_method: DescentMethodsEnum::QuasiNewton(QuasiNewtonMethods::DFP),
        descent_method: DescentMethodsEnum::QuasiNewton(QuasiNewtonMethods::DFPMoreStable),
        optimization_error_margin: 0.1,
        differentiation_step_size: 0.000000001,
        line_search_initial_step_size: 0.5,
        linesearch_error_margin: 0.00001,
    };

    let (function_evaluations, itterations, mut plot, convergence) = assignment_2(optimization_settings);

    println!("Number of itterations: {}, number of function evaluations: {}, last error {}", itterations, function_evaluations, convergence.last().unwrap());
    plot.set_title("Steepest descent").set_x_label("x_1").set_y_label("x_2");
    plot.plotting_settings.axis_equal_length = true;
    plot.plotting_settings.show_grid_major = true;
    plot.plotting_settings.color_map_line = PlotBuilderColorMaps::Viridis;
    // plot.to_plotters_processor().SVG_to_file("plots/SD_test.svg");

    let mut convergence_plot = PlotBuilder::new();
    for index in  0..convergence.len() -1 {
        convergence_plot.add_simple_2d_line(&vec![index as f64, (index+1) as f64], &vec![convergence[index],convergence[index+1] ]);
    }
    convergence_plot.set_y_axis_type(PlotAxisScaling::Log);
    convergence_plot
        .set_title("Convergence")
        .set_x_label("Itteration")
        .set_y_label("|F ∇|")
        .scale_plot(2.);
    convergence_plot.plotting_settings.axis_equal_length = false;
    convergence_plot.plotting_settings.color_map_line = PlotBuilderColorMaps::Viridis;
    convergence_plot.plotting_settings.show_grid_major = true;
    convergence_plot.plotting_settings.show_y_grid_minor = true;
    convergence_plot.to_plotters_processor().SVG_to_file("plots/convergence_test.svg");
}

fn cost_function_slope(x: Vec<f64>) -> f64 {
    (x[0]).powi(2) + x[1].powi(2) 
}

struct OptimizationSettings {
    initial_point: PosNDof<f64>,
    extrapolation_method: LineSearchExtrapolationMethods,
    interpolation_method: LineSearchInterpolationMethods,
    descent_method: DescentMethodsEnum,
    optimization_error_margin: f64,
    differentiation_step_size: f64,
    line_search_initial_step_size: f64,
    linesearch_error_margin: f64,
}

// fn roosenbrook_interactive_bfgs(plot_settings: InteractivePlotSettings) -> PlotBuilder<f64> {
fn assignment_2(optimization_settings: OptimizationSettings) -> (usize, usize, PlotBuilder<f64>, Vec<f64>) {

    let mut plot = PlotBuilder::new();

    let mut x_step = Vec::<f64>::new();
    let mut y_step = Vec::<f64>::new();
    let mut slope_length = Vec::<f64>::new();

    let mut total_number_function_evals: usize = 0;

    let cost_function: Rc<Box<dyn Fn(Vec<f64>) -> f64>> = Rc::new( Box::new(|x: Vec<f64>| {
        // get_cost_function_springs(x)
        get_cost_function_roosenbruck(x)
    }));

    plot.add_contour_plot_fn(cost_function.clone());

    let mut search_direction = DescentMethods::new_unconstrained(
            cost_function.clone(),
            &optimization_settings.initial_point,
            optimization_settings.differentiation_step_size
        )
        .descent_method(DescentMethodsEnum::SteepestDescent);

    x_step.push(search_direction.search_line().base_point().x[0]);
    y_step.push(search_direction.search_line().base_point().x[1]);
    x_step.push(search_direction.search_line().base_point().x[0]);
    y_step.push(search_direction.search_line().base_point().x[1]);

    let mut index: usize = 0;
    while search_direction.local_gradient().length() > optimization_settings.optimization_error_margin { 
        index += 1;

        slope_length.push(search_direction.local_gradient().length());

        let line_searcher_uninitialized = LineSearchMethods::new_with_search_line_and_initial_stepsize(
            cost_function.clone(), 
            &search_direction.search_line(), 
            optimization_settings.line_search_initial_step_size, 
            optimization_settings.linesearch_error_margin
        );

        let line_searcher_bracked: LineSearchMethods<Bracketed>;
        match optimization_settings.extrapolation_method {
            LineSearchExtrapolationMethods::GoldenRuleExtrapolation => {
                line_searcher_bracked = line_searcher_uninitialized.bracket_extrapolation_method(LineSearchExtrapolationMethods::GoldenRuleExtrapolation);
            },
        }
        
        let mut line_searcher = line_searcher_bracked.interpolation_method(optimization_settings.interpolation_method.clone());

        line_searcher.itterate_untill_convergence();

        let search_point = line_searcher.bracket_averaged_variables();

        x_step.remove(0);
        y_step.remove(0);
        x_step.push(search_point.x[0]);
        y_step.push(search_point.x[1]);
        plot.add_simple_2d_line(&x_step, &y_step);

        search_direction = search_direction.update_search_point_and_local_gradient(search_point)
            .descent_method(optimization_settings.descent_method.clone());

        total_number_function_evals += line_searcher.used_function_evals();
        if index == 100 {
            break;
        }
    }
    slope_length.push(search_direction.local_gradient().length());
    total_number_function_evals += search_direction.used_function_evals;
    (total_number_function_evals, index, plot, slope_length)
    }

    // fn assignment_2_simple_plots()  {
    //     // let a = 1.;
    //     // let b = 100.;
    //     let plot_settings = InteractivePlotSettings {
    //         x_range: -3.5 .. 2.,
    //         y_range: -3. .. 2.5,
    //         plot_height: 500,
    //         plot_width: 500,
    //         some_input_number: 1,
    //     };
    
    //     let mut total_number_function_evals: usize = 0;
    
    //     let cost_function: Rc<Box<dyn Fn(Vec<f64>) -> f64>> = Rc::new( Box::new(|x: Vec<f64>| {
    //         get_cost_function_springs(x)
    //         // cost_function_slope(x)
    //     }));
    
    
    //     let mut search_point = PosNDof::new(vec![0.5, -0.25]);
    //     let error_margin_line_search = 0.0000001;
    //     let error_margin_optimization = 0.1;
    //     let initial_step_size = 0.01;
    
    //     let mut search_direction = DescentMethods::new_unconstrained(cost_function.clone(),&search_point).use_steepest_descent();
    
    //                 let mut x_points = Vec::<f64>::new();
    //                 let mut y_points = Vec::<f64>::new();
    
    //                 x_points.push(search_point.x[0]);
    //                 y_points.push(search_point.x[1]);
    //                 x_points.push(search_point.x[0]);
    //                 y_points.push(search_point.x[1]);
    
    
    //                 let mut plot = PlotBuilder::new();
    
    //                 let x_values = f64::linspace(plot_settings.x_range.start, plot_settings.x_range.end, 200);
    //                 let y_values = f64::linspace(plot_settings.y_range.start, plot_settings.y_range.end, 200);
    //                 let mut z_values = vec![vec![0.; y_values.len()];x_values.len()];
    //                 for index_x in 0..x_values.len() {
    //                     for index_y in 0..y_values.len() {
    //                         z_values[index_x][index_y] = (cost_function.clone())(vec![x_values[index_x],y_values[index_y]])
    //                     }
    //                 }
    
    //                 let heatmap = Surface3d::new_contour_fn(cost_function.clone());
    //                 plot.add_contour(heatmap.clone());
    //                 // let heatmap = Surface3d::new_surface_fn(cost_function.clone());
    //                 // plot.add_surface_3d(heatmap.clone());
    
    
    //                 plot.set_plot_size(plot_settings.plot_width, plot_settings.plot_height);
    //                 plot.plotting_settings.axis_equal_length = true;
    //                 // plot.set_x_range(plot_settings.x_range).set_y_range(plot_settings.y_range);
    
    //                 // plot.add_simple_2d_line(&x_points, &y_points);
    
    //     let mut index: usize = 0;
    //     let follow_gradient = false;
    //     while search_direction.local_gradient().length() > error_margin_optimization { 
    //         index += 1;
    
                
    //         // if search_direction.search_line().length() < 1. {
    //         //     search_point = search_direction.search_line().end_position();
    //         // } else {
    //         let mut line_searcher = LineSearchMethods::new_with_search_line_and_initial_stepsize(
    //             cost_function.clone(), 
    //             &search_direction.search_line(), 
    //             initial_step_size, 
    //             error_margin_line_search)
    //             .find_bracket_golden_rule_expansion()
    //             .use_golden_section_interpolation();
    
    //         line_searcher.itterate_untill_convergence();
    //         search_point = line_searcher.bracket_averaged_variables();
    
    //         total_number_function_evals += line_searcher.used_function_evals;
    
    
    //                     x_points.remove(0);
    //                     x_points.push(search_point.x[0]);
    //                     y_points.remove(0);
    //                     y_points.push(search_point.x[1]);
    
    //                     plot.add_simple_2d_line(&x_points, &y_points);
    //                     plot.plotting_settings.color_map_line = PlotBuilderColorMaps::Viridis;
            
    //         search_direction = search_direction.update_search_point_and_local_gradient(search_point)
    //             .use_steepest_descent();
    
    
    //             if index % 10 == 0 {
    //                 println!("loop: {}", index);
    //             }
    
    //         if index == 1000 {
    //             break;
    //         }
    //     }
    
    //     total_number_function_evals += search_direction.used_function_evals;
    
    //     println!("steps used: {}", index);
    //                     plot.plotting_settings.title_font_size = 40;
    //                     plot.add_simple_2d_line(&x_points, &y_points);
    //                     // plot.plotting_settings.show_x_grid_minor = true;
    //                     // plot.plotting_settings.show_y_grid_minor = true;
    //                     // plot.plotting_settings.show_grid_major = true;
    //                     plot.set_title("Golden section SD").set_x_label("x_1").set_y_label("x_2").scale_plot(4.);
                        
    //                     plot.to_plotters_processor().SVG_to_file("plots/golden_section_sd.svg");
      
    //     }

// fn quadratic() {
//     let cost_function: Rc<Box<dyn Fn(Vec<f64>) -> f64>> = Rc::new(Box::new(|x:Vec<f64>| 0.01* x[0].powi(2)));

//     let x_curve = <f64>::linspace(-10., 10., 1000);
//     let y_curve = x_curve.iter().map(|x| cost_function(vec![*x])).collect::<Vec<f64>>();

//     let start_point = PosNDof::new(vec![-10.]);
//     let initial_step_size = 0.01;
//     let error_margin_line_search = 0.01;

//     let mut x_values = Vec::<f64>::new();
//     let mut y_values = Vec::<f64>::new();
//     let mut plot = PlotBuilder::new();
//     plot.add_simple_2d_line(&x_curve,&y_curve);
//     plot.set_x_range(-10. .. 10.).set_y_range(0. .. 1.).set_plot_size(1000, 1000);
//     // plot.set_x_range(-5. .. 5.).set_y_range(-0.25 .. 0.25);

//     let search_line = DescentMethods::new(cost_function.clone(), &start_point).use_steepest_descent();

//     let mut line_searcher = LineSearchMethods::new_with_search_line_and_initial_stepsize(
//         cost_function.clone(), 
//         &search_line.search_line(), 
//         initial_step_size, 
//         error_margin_line_search)
//         .find_bracket_golden_rule_expansion()
//         .use_quadratic_interpolation();

//     plot.add_simple_2d_line(
//         &vec![line_searcher.coordinates_of_lower_bound()[0], line_searcher.coordinates_of_upper_bound()[0]], 
//         &vec![line_searcher.cost(0.), line_searcher.cost(1.)]
//     );
//     plot.clone().to_plotters_processor().SVG_to_file("quadr_test.svg");
    
//     let mut index: usize = 0;
//     while line_searcher.is_completed().not() {
//         index += 1;
//         line_searcher.itterate();

//         plot.add_simple_2d_line(
//             &vec![line_searcher.coordinates_of_lower_bound()[0], line_searcher.search_line.volatile().scale(line_searcher.central_point.unwrap()).end_position().x[0], line_searcher.coordinates_of_upper_bound()[0]], 
//             &vec![line_searcher.cost(0.), line_searcher.cost_central_point.unwrap(), line_searcher.cost(1.)]
//         );

//         plot.clone().to_plotters_processor().SVG_to_file("quadr_test.svg");

//         let mut input = String::new();
//         io::stdin().read_line(&mut input).unwrap();

//         if index == 7 {
//             println!("debug");
//             // io::stdin().read_line(&mut input).unwrap();
//         }

//         if index == 10 {
//             break;
//         }
//     }
    
//     println!("itter stopped at {} itterations", index);


// }



// fn roosenbrook_interactive(plot_settings: InteractivePlotSettings) -> PlotBuilder<f64> {
//     // let a = 1.;
//     // let b = 100.;

//     let cost_function: Rc<Box<dyn Fn(Vec<f64>) -> f64>> = Rc::new( Box::new(|x: Vec<f64>| {
//         let a = 1.;
//         let b = 100.;
//         (a - x[0]).powi(2) + b*(x[1] - x[0].powi(2)).powi(2)}));

//     // let cloned = cost(/*= Vec<f64> */).clone();
//     // let cost_function = get_cost_function_roosenbruck();

//     let mut search_point = PosNDof::new(vec![-2., 3.]);
//     let error_margin_line_search = 0.00001;
//     let error_margin_optimization = 0.01;
//     let initial_step_size = 0.0001;

//     let mut search_direction = DescentMethods::new_unconstrained(cost_function.clone(),&search_point).use_steepest_descent();

//                 let mut x_points = Vec::<f64>::new();
//                 let mut y_points = Vec::<f64>::new();

//                 x_points.push(search_point.x[0]);
//                 y_points.push(search_point.x[1]);


//                 let mut plot = PlotBuilder::new();
//                 let mut plot2 = PlotBuilder::new();

//                 let x_values = f64::linspace(-2., 2., 200);
//                 let y_values = f64::linspace(-3., 3., 200);
//                 let mut z_values = vec![vec![0.; y_values.len()];x_values.len()];
//                 for index_x in 0..x_values.len() {
//                     for index_y in 0..y_values.len() {
//                         z_values[index_x][index_y] = (cost_function.clone())(vec![x_values[index_x],y_values[index_y]])
//                     }
//                 }

//                 let heatmap = Surface3d::new_surface_xyz(&x_values, &y_values, &z_values);
//                 plot.add_surface_3d(heatmap.clone());
//                 plot.set_plot_size(500, 500);

//                 plot.add_simple_2d_line(&x_points, &y_points);
//                 plot2.add_simple_2d_line_y_only(&x_points);
//                 plot2.add_simple_2d_line_y_only(&y_points);

//                 // plot.clone().to_plotters_processor().SVG_to_file("roosenbrok.svg");


//     let mut cost = (cost_function.clone())(search_point.x);
//     let mut index: usize = 0;
//     while search_direction.local_gradient().length() > error_margin_optimization {
//     // for index in 0..200 {
//         // println!("loop {}, search direction length = {}, cost here (1000x) = {:.2}, x = {:.2}, y = {:.2}", index, search_direction.search_line.clone().length(), cost*1000., search_direction.search_line.base_point_vec()[0], search_direction.search_line.base_point_vec()[1]);
//         index += 1;
//         let mut line_searcher = LineSearchMethods::new_with_search_line_and_initial_stepsize(
//             cost_function.clone(), 
//             &search_direction.search_line(), 
//             initial_step_size, 
//             error_margin_line_search)
//             .find_bracket_golden_rule_expansion()
//             .use_golden_section_interpolation();

//         line_searcher.itterate_untill_convergence();

//         search_point = line_searcher.bracket_averaged_variables();

//                     cost = (cost_function.clone())(search_point.clone().x);
//                     x_points.push(search_point.x[0]);
//                     y_points.push(search_point.x[1]);

//                     plot.add_surface_3d(heatmap.clone());

//                     plot.add_simple_2d_line(&x_points, &y_points);
//                     plot2.add_simple_2d_line_y_only(&x_points);
//                     plot2.add_simple_2d_line_y_only(&y_points);
                
//                     // plot.clone().to_plotters_processor().SVG_to_file("roosenbrok.svg");
//                     // plot2.clone().to_plotters_processor().SVG_to_file("roosenbrokxyinteractive.svg");

//         search_direction = search_direction.update_search_point_and_local_gradient(search_point)
//             .use_conjugate_descent(ConjugateGradientMethods::PolakRibiere);


//         if index % 100 == 0 {
//             println!("step: {}, cost {}, search_direction.search_line.length() {}", index, cost, search_direction.local_gradient().length())
//         }

//         if index > 500 {
//             break;
//         }
//     }


//     println!("steps used: {}", index);



//                     plot.add_simple_2d_line(&x_points, &y_points);

//                     plot2.add_simple_2d_line_y_only(&x_points);
//                     plot2.add_simple_2d_line_y_only(&y_points);

//                     let mut plot_zoomed = plot.clone();
//                     let x_values = f64::linspace(-0.5, 0.5, 200);
//                     let y_values = f64::linspace(-0.05, 0.2, 200);
//                     let mut z_values = vec![vec![0.; y_values.len()];x_values.len()];
//                     for index_x in 0..x_values.len() {
//                         for index_y in 0..y_values.len() {
//                             z_values[index_x][index_y] = (cost_function.clone())(vec![x_values[index_x],y_values[index_y]])
//                         }
//                     }
//                     let heatmap = Surface3d::new_surface_xyz(&x_values, &y_values, &z_values);
//                     plot_zoomed.add_surface_3d(heatmap.clone());
//                     let mut plot_zoomed_finish = plot_zoomed.clone();
//                     plot_zoomed_finish.set_x_range(0.9 ..1.1).set_y_range(0.9 .. 1.1).set_plot_size(1000, 1000);
//                     plot_zoomed.set_x_range(-0.5 .. 0.5).set_y_range(-0.05 .. 0.2).set_plot_size(1000, 1000);
//                     // plot_zoomed.to_plotters_processor().SVG_to_file("zoomedroosen.svg");
//                     // plot_zoomed_finish.to_plotters_processor().SVG_to_file("zoomedroosenfinsih.svg");

//                     // plot.clone().to_plotters_processor().SVG_to_file("roosenbrok.svg");
//                     // plot2.clone().to_plotters_processor().SVG_to_file("roosenbrokxy.svg");

//                     plot
//                 }


// fn roosenbrook() {
//     // let a = 1.;
//     // let b = 100.;

//     let cost_function: Rc<Box<dyn Fn(Vec<f64>) -> f64>> = Rc::new( Box::new(|x: Vec<f64>| {
//         let a = 1.;
//         let b = 100.;
//         (a - x[0]).powi(2) + b*(x[1] - x[0].powi(2)).powi(2)}));

//     // let cloned = cost(/*= Vec<f64> */).clone();
//     // let cost_function = get_cost_function_roosenbruck();

//     let mut search_point = PosNDof::new(vec![-2., 3.]);
//     let error_margin_line_search = 0.00001;
//     let error_margin_optimization = 0.01;
//     let initial_step_size = 0.0001;

//     let mut search_direction = DescentMethods::steepest_descent(cost_function.clone(), &search_point);

//                 let mut x_points = Vec::<f64>::new();
//                 let mut y_points = Vec::<f64>::new();

//                 x_points.push(search_point.x[0]);
//                 y_points.push(search_point.x[1]);


//                 let mut plot = PlotBuilder::new();
//                 let mut plot2 = PlotBuilder::new();

//                 let x_values = f64::linspace(-2., 2., 200);
//                 let y_values = f64::linspace(-3., 3., 200);
//                 let mut z_values = vec![vec![0.; y_values.len()];x_values.len()];
//                 for index_x in 0..x_values.len() {
//                     for index_y in 0..y_values.len() {
//                         z_values[index_x][index_y] = (cost_function.clone())(vec![x_values[index_x],y_values[index_y]])
//                     }
//                 }

//                 let heatmap = Surface3d::new(&x_values, &y_values, &z_values);
//                 plot.add_surface_3d(heatmap.clone()).set_plot_size(1000, 1000);

//                 plot.add_simple_2d_line(&x_points, &y_points);
//                 plot2.add_simple_2d_line_y_only(&x_points);
//                 plot2.add_simple_2d_line_y_only(&y_points);

//                 plot.clone().to_plotters_processor().SVG_to_file("roosenbrok.svg");


//     let mut cost = (cost_function.clone())(search_point.x);
//     let mut index: usize = 0;
//     while search_direction.gradient.length() > error_margin_optimization {
//     // for index in 0..200 {
//         // println!("loop {}, search direction length = {}, cost here (1000x) = {:.2}, x = {:.2}, y = {:.2}", index, search_direction.search_line.clone().length(), cost*1000., search_direction.search_line.base_point_vec()[0], search_direction.search_line.base_point_vec()[1]);
//         index += 1;
//         let mut line_searcher = LineSearchMethods::<UnkownBracketPointOnly>::new_with_search_line_and_initial_stepsize(
//             cost_function.clone(), 
//             &search_direction.search_line, 
//             initial_step_size, 
//             error_margin_line_search)
//             .find_bracket_golden_rule_expansion()
//             .use_golden_section_interpolation();

//         line_searcher.itterate_untill_convergence();

//         search_point = line_searcher.bracket_averaged_variables();

//                     cost = (cost_function.clone())(search_point.clone().x);
//                     x_points.push(search_point.x[0]);
//                     y_points.push(search_point.x[1]);

//                     plot.add_surface_3d(heatmap.clone());

//                     plot.add_simple_2d_line(&x_points, &y_points);
//                     plot2.add_simple_2d_line_y_only(&x_points);
//                     plot2.add_simple_2d_line_y_only(&y_points);
                
//                     plot.clone().to_plotters_processor().SVG_to_file("roosenbrok.svg");
//                     plot2.clone().to_plotters_processor().SVG_to_file("roosenbrokxy.svg");

//         // search_direction = DescentMethods::conjugate_descent(
//         //     cost_function.clone(), 
//         //     search_point,
//         // search_direction,
//         // ConjugateGradientMethods::PolakRibiere
//         //     );
//         search_direction = DescentMethods::steepest_descent(cost_function.clone(), &search_point);

//         if index % 100 == 0 {
//             println!("step: {}, cost {}, search_direction.search_line.length() {}", index, cost, search_direction.gradient.length())
//         }

//         if index > 500 {
//             break;
//         }
//     }


//     println!("steps used: {}", index);



//                     plot.add_simple_2d_line(&x_points, &y_points);

//                     plot2.add_simple_2d_line_y_only(&x_points);
//                     plot2.add_simple_2d_line_y_only(&y_points);

//                     let mut plot_zoomed = plot.clone();
//                     let x_values = f64::linspace(-0.5, 0.5, 200);
//                     let y_values = f64::linspace(-0.05, 0.2, 200);
//                     let mut z_values = vec![vec![0.; y_values.len()];x_values.len()];
//                     for index_x in 0..x_values.len() {
//                         for index_y in 0..y_values.len() {
//                             z_values[index_x][index_y] = (cost_function.clone())(vec![x_values[index_x],y_values[index_y]])
//                         }
//                     }
//                     let heatmap = Surface3d::new(&x_values, &y_values, &z_values);
//                     plot_zoomed.add_surface_3d(heatmap.clone());
//                     let mut plot_zoomed_finish = plot_zoomed.clone();
//                     plot_zoomed_finish.set_x_range(0.9 ..1.1).set_y_range(0.9 .. 1.1).set_plot_size(1000, 1000);
//                     plot_zoomed.set_x_range(-0.5 .. 0.5).set_y_range(-0.05 .. 0.2).set_plot_size(1000, 1000);
//                     plot_zoomed.to_plotters_processor().SVG_to_file("zoomedroosen.svg");
//                     plot_zoomed_finish.to_plotters_processor().SVG_to_file("zoomedroosenfinsih.svg");

//                     plot.to_plotters_processor().SVG_to_file("roosenbrok.svg");
//                     plot2.clone().to_plotters_processor().SVG_to_file("roosenbrokxy.svg");
// }

// fn heatmap_test() {
//     let function = get_cost_function_roosenbruck();
//     let x_values = f64::linspace(-2., 2., 200);
//     let y_values = f64::linspace(-3., 3., 200);
//     let mut z_values = vec![vec![0.; y_values.len()];x_values.len()];
//     for index_x in 0..x_values.len() {
//         for index_y in 0..y_values.len() {
//             z_values[index_x][index_y] = (function)(vec![x_values[index_x],y_values[index_y]])
//         }
//     }
//     let mut plot = PlotBuilder::<f64>::new();
//     let heatmap = Surface3d::new(&x_values, &y_values, &z_values);
//     plot.add_surface_3d(heatmap);

//     plot.add_simple_2d_line(&vec![-2., 0., 2.], &vec![-3., 3., -3.]);

//     plot.to_plotters_processor().SVG_to_file("test_heat.svg")

// }

// fn line_search() {
//     let cost_function = get_cost_function_roosenbruck();
//     let search_point = PosNDof::new(vec![15.]);
//     let error_margin = 0.0001;
//     let initial_step_size = 1.;

//     let mut plot = PlotBuilder::new();
//     let mut x_values = Vec::<f64>::new();
//     let mut y_values = Vec::<f64>::new();

//     {
//         let search_direction = SteepestDecent::steepest_decent(
//             cost_function, 
//             &search_point
//         );

//         let cost_function = get_cost_function_roosenbruck();

//         let mut find_line_minimum = GoldenLineSearch::new_find_bracket_from_initial_guess_and_initial_step_size(
//             cost_function,
//             &search_direction.search_line,
//             initial_step_size,
//             error_margin,
//         );

//         x_values.push(find_line_minimum.coordinates_of_lower_bound()[0]);
//         x_values.push(find_line_minimum.coordinates_of_upper_bound()[0]);
//         y_values.push(find_line_minimum.cost(0.));
//         y_values.push(find_line_minimum.cost(1.));

//         while find_line_minimum.is_completed().not() {
//             find_line_minimum.itterate();

//             x_values.push(find_line_minimum.coordinates_of_lower_bound()[0]);
//             x_values.push(find_line_minimum.coordinates_of_upper_bound()[0]);
//             y_values.push(find_line_minimum.cost(0.));
//             y_values.push(find_line_minimum.cost(1.));
//         }
//     }

//     plot.add_simple_2d_line(&x_values, &y_values);

//     let cost_function = get_cost_function_roosenbruck();
//     let x_curve = f64::linspace(-2., 2., 200);
//     let y_curve = f64::linspace(-3., 3., 200);
//     let mut line_curve = Line2d::new(&x_curve, &y_curve);
//     line_curve.set_color(RGBAColor(0, 0, 0, 1.));
//     plot.add_2d_line(&line_curve);

//     plot.to_plotters_processor().SVG_to_file("test.svg")


// }

fn get_cost_function_springs(x: Vec<f64>) -> f64 {
    let position_node_1 = PosNDof::new(vec![0., l0]);
    let position_node_2 = PosNDof::new(vec![0., -l0]);


    let dl = |position_node: PosNDof<f64>, x: Vec<f64>| {
        ((x[0] - position_node.x[0]).powi(2) + (x[1] - position_node.x[1]).powi(2)).sqrt() - l0
    };

    let cost = 0.5 * k1 * dl(position_node_1.clone(), x.clone()).powi(2)
    + 0.5 * k2 * dl(position_node_2.clone(), x.clone()).powi(2)
    - p1 * x[0]
    - p2 * x[1]
    ;

    cost
}

// trait NewTrait: Fn(Vec<f64>) -> f64 + Clone + Sized {}

fn get_cost_function_roosenbruck(x: Vec<f64>) -> f64 {
    let a = 1.;
    let b = 100.;

    (a - x[0]).powi(2) + b*(x[1] - x[0].powi(2)).powi(2)

}

// fn get_cost_function_simple_parabola() -> Box<dyn Fn(Vec<f64>) -> f64> {
//     let cost_function = Box::new(|x:Vec<f64>| 0.01* x[0].powi(2));

//     cost_function
// }