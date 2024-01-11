use crate::{*};

pub type CostFunction = Rc<Box<dyn Fn(&Vec<f64>) -> ObjectiveResult + 'static >>;
pub type ConstraintFunction = Rc<Box<dyn Fn(&Vec<f64>) -> f64 + 'static >>;
pub type AlgebraicDerivative = Rc<Box<dyn Fn(&Vec<f64>) -> Vec<f64> + 'static >>;

pub struct ObjectiveResult {
    pub function_cost: f64,
    pub model_constraints: Vec<InequalityConstraintResult>,
}

#[macro_export]
macro_rules! make_cost_function {
    ($function: ident) => {
        {
            let result: CostFunction = Rc::new(Box::new(|x: &Vec<f64>| {$function(&x)}));
            result
        }
    };
}

#[macro_export]
macro_rules! make_inequality_constraint_function {
    ($function: ident) => {
        {
            let result: ConstraintFunction = Rc::new(Box::new(|x: &Vec<f64>| {$function(&x)}));
            result
        }
    };
}

#[derive(Debug, Clone, Copy)]
pub enum DifferentiationRule {
    CentralDifference,
    OneSided
}

#[derive(Clone)]
pub struct ObjectiveFunction {
    pub constraint_function: CostFunction,
    pub algebraic_derivative_supplied: bool,
    pub algebraic_derivative: Option<AlgebraicDerivative>,
}

#[derive(Clone)]
pub struct InequalityConstraintResult {
    pub value: f64,
    pub algebraic_derivative_supplied: bool,
    pub algebraic_derivative: Option<AlgebraicDerivative>,
}

#[derive(Clone)]
pub struct InequalityConstraint {
    pub constraint_function: ConstraintFunction,
    pub algebraic_derivative_supplied: bool,
    pub algebraic_derivative: Option<AlgebraicDerivative>,
}

impl InequalityConstraint {
    pub fn new(constraint_function: ConstraintFunction) -> Self {
        Self {
            constraint_function,
            algebraic_derivative_supplied: false ,
            algebraic_derivative: Option::None,
        }
    }

    pub fn new_with_derivative(constraint_function: ConstraintFunction, algebraic_derivative: AlgebraicDerivative) -> Self {
        Self {
            constraint_function,
            algebraic_derivative_supplied: true,
            algebraic_derivative: Option::Some(algebraic_derivative),
        }
    }
}

#[derive(Clone)]
pub struct Optimizer {
    pub descent_controller: DescentController,
    pub plot_contour: bool,
    pub plot_constraints: bool,
}

impl Optimizer {
    pub fn new(
        descent_controller: DescentController,
    ) -> Self {
        let plot_constraints: bool;
        match descent_controller.optimization_settings.constraint_rules.constrained {
            OptimizationType::Unconstrained => {
                plot_constraints = false
            },
            _=> {
                plot_constraints = true
            },
        }
        Optimizer {
            descent_controller,
            plot_contour: true,
            plot_constraints,
        }
    }
}

impl Optimizer {
    pub fn run_optimization(&mut self) -> &mut Self {
        self.descent_controller.run_optimization();
        self
    }

    pub fn plot_coordinates(&mut self, path: &str) -> &mut Self {
        
        match self.descent_controller.optimization_settings.logging_settings.track_variable_history {
            true => {},
            false => {panic!("Can not make variable history plot! Variables have not been tracked.");}
        }

        let coordinates_log = self.descent_controller.descent_logger.coordinates_history.clone();

        let mut plot = PlotBuilder::new();
        plot.scale_plot(4.);

        ;
        {
            match self.descent_controller.n_dof() {
                1 => {
                    plot.add_simple_2d_line_y_only(&coordinates_log[0])
                        .set_x_label("# Iteration")
                        .set_y_label("x");
                    plot.plotting_settings.axis_equal_length = true;
                },
                2 => {
                    for line_partition in 0 .. coordinates_log[0].len() -1 {
                        let line_2d = Line2d::new(&vec![coordinates_log[0][line_partition], coordinates_log[0][line_partition+1]],&vec![coordinates_log[1][line_partition],coordinates_log[1][line_partition+1]] );
                        plot.add_2d_line(&line_2d);
                    }
                    // let line_2d = Line2d::new(&coordinates_log[0],&coordinates_log[1]);
                    // plot.add_2d_line(&line_2d);
                    plot.set_x_label("X_1")
                        .set_y_label("X_2");
                    plot.plotting_settings.axis_equal_length = true;

                    if self.plot_contour {
                        let cost_controller_plot = self.descent_controller.line_searcher.cost_controller.clone();
                        let mut contour_plot = Surface3d::new_contour_fn(
                            Rc::new(Box::new(move |x| CostController {
                                cost_function: cost_controller_plot.cost_function.clone(),
                                cost_function_normalization_factor: cost_controller_plot.cost_function_normalization_factor.clone(),
                                constraint_equality_functions: cost_controller_plot.constraint_equality_functions.clone(),
                                constraint_inequality_functions: cost_controller_plot.constraint_inequality_functions.clone(),
                                number_cost_function_evaluation: cost_controller_plot.number_cost_function_evaluation.clone(),
                                number_constraint_function_evaluations: cost_controller_plot.number_constraint_function_evaluations.clone(),
                                constraint_rules: cost_controller_plot.constraint_rules.clone(),
                                optimization_settings: cost_controller_plot.optimization_settings.clone(),
                                n_constraints_objective_result: cost_controller_plot.n_constraints_objective_result.clone(),
                            }.cost_at_x(&x)))
                        );

                        // contour_plot.plot_zeroth_contour = Option::Some(false);

                        plot.add_contour(contour_plot);


                        if self.plot_constraints {
                            

                            for constraint_index in 0..self.descent_controller.line_searcher.cost_controller.n_constraints(){
                                let cost_controller_plot = self.descent_controller.line_searcher.cost_controller.clone();
                                
                                let mut contour_plot = Surface3d::new_contour_fn(
                                    Rc::new(Box::new(move |x| CostController {
                                        cost_function: cost_controller_plot.cost_function.clone(),
                                        cost_function_normalization_factor: cost_controller_plot.cost_function_normalization_factor.clone(),
                                        constraint_equality_functions: cost_controller_plot.constraint_equality_functions.clone(),
                                        constraint_inequality_functions: cost_controller_plot.constraint_inequality_functions.clone(),
                                        number_cost_function_evaluation: cost_controller_plot.number_cost_function_evaluation.clone(),
                                        number_constraint_function_evaluations: cost_controller_plot.number_constraint_function_evaluations.clone(),
                                        constraint_rules: cost_controller_plot.constraint_rules.clone(),
                                        optimization_settings: cost_controller_plot.optimization_settings.clone(),
                                        n_constraints_objective_result: cost_controller_plot.n_constraints_objective_result.clone(),
                                    }.inequality_contraint_value(constraint_index, &x)))
                                );
                                contour_plot.use_constraint_filled_preset();

                                plot.add_contour(contour_plot);
                                

                            }
                        }
                        
                    }
                },
                _ => {
                    for coordinate in coordinates_log {
                        plot.add_simple_2d_line_y_only(&coordinate);
                    }
                    plot.set_x_label("# Iteration")
                        .set_y_label("x");
                }
            }

            // plot.set_x_range(10. .. 15.).set_y_range(18. .. 20.);
                                plot.set_x_range(-1.1 .. 2.9).set_y_range(-0.6 .. 3.4);
                                plot.plotting_settings.contour_n_points = 175;

            plot.plotting_settings.color_map_line = PlotBuilderColorMaps::Viridis(1.);
            plot.to_plotters_processor().bitmap_to_file(path);
        }
        self
    }

    pub fn plot_convergence(&mut self, path: &str) -> &mut Self {
        match self.descent_controller.optimization_settings.logging_settings.track_convergence_history {
            true => {},
            false => {panic!("Can not make convergence plot! Convergence has not been tracked.");}
        }

        let convergence_log = self.descent_controller.descent_logger.convergence_history.clone();

        let mut plot = PlotBuilder::new();
        plot.scale_plot(4.);

        for line_partition in 0 .. convergence_log.len() -1 {
            let simple_line_partition = vec![convergence_log[line_partition], convergence_log[line_partition+1]];
            let x_vlues = vec![line_partition as f64, (line_partition + 1) as f64];
            plot.add_simple_2d_line(&x_vlues, &simple_line_partition);
        }
        // plot.add_simple_2d_line_y_only(&convergence_log);

        plot.set_y_axis_type(PlotAxisScaling::Log);

        plot.set_x_label("# Iteration")
            .set_y_label("|Difergence(F)|");
        plot.plotting_settings.show_grid_major = true;

        plot.plotting_settings.color_map_line = PlotBuilderColorMaps::Viridis(1.);
        plot.to_plotters_processor().bitmap_to_file(path);

        self
    }
}