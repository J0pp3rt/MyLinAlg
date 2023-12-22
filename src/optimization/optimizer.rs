use crate::{*};

pub type CostFunction = Rc<Box<dyn Fn(&Vec<f64>) -> f64 + 'static >>;
pub type AlgebraicDerivative = Rc<Box<dyn Fn(&Vec<f64>) -> Vec<f64> + 'static >>;

#[macro_export]
macro_rules! make_cost_function {
    ($function: ident) => {
        Rc::new(Box::new(|x: &Vec<f64>| {$function(&x)}))
    };
}

#[derive(Debug, Clone, Copy)]
pub enum DifferentiationRule {
    CentralDifference,
    OneSided
}

#[derive(Clone)]
pub struct InequalityConstraint {
    pub constraint_function: CostFunction,
    pub algebraic_derivative_supplied: bool,
    pub algebraic_derivative: Option<AlgebraicDerivative>,
}

impl InequalityConstraint {
    pub fn new(constraint_function: CostFunction) -> Self {
        Self {
            constraint_function,
            algebraic_derivative_supplied: false ,
            algebraic_derivative: Option::None,
        }
    }

    pub fn new_with_derivative(constraint_function: CostFunction, algebraic_derivative: AlgebraicDerivative) -> Self {
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
    plot_contour: bool,
    plot_constraints: bool,
}

impl Optimizer {
    pub fn new(
        descent_controller: DescentController,
    ) -> Self {
        let plot_constraints: bool;
        match descent_controller.optimization_settings.constraint_rules.constrained {
            OptimizationType::Constrained => {
                plot_constraints = true
            },
            OptimizationType::Unconstrained => {
                plot_constraints = false
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
                    let line_2d = Line2d::new(&coordinates_log[0],&coordinates_log[1]);
                    plot.add_2d_line(&line_2d)
                        .set_x_label("X_1")
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
                            }.cost_at_x(&x)))
                        );

                        plot.add_contour(contour_plot);


                        if self.plot_constraints {
                            

                            for constraint_index in 0..self.descent_controller.line_searcher.cost_controller.constraint_inequality_functions.len(){
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
                                    }.inequality_contraint_value(constraint_index, &x)))
                                );
                                contour_plot.use_constraint_contour_preset();

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

        plot.add_simple_2d_line_y_only(&convergence_log);

        plot.set_y_axis_type(PlotAxisScaling::Log);

        plot.set_x_label("# Iteration")
            .set_y_label("|Difergence(F)|");
        plot.plotting_settings.show_grid_major = true;

        plot.to_plotters_processor().bitmap_to_file(path);

        self
    }
}