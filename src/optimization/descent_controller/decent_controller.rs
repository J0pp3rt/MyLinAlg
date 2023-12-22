use crate::{*};
#[derive(Debug, Clone, Copy)]
pub enum DescentMethods {
    SteepestDescent,
    ConjugateDescent(ConjugateGradient),
    QuasiNewton(QuasiNewton)
}
#[derive(Debug, Clone, Copy)]
pub enum ConjugateGradient {
    FletcherReeves,
    PolakRibiere
}
#[derive(Debug, Clone, Copy)]
pub enum QuasiNewton {
    BFGS,
    BFGSBeforeInverse,
    BFGSBeforeInverseMoreStable,
    DFP,
    DFPMoreStable
}

#[derive(Debug, Clone, Copy)]
pub enum ConstraintMerit {
    ExteriorQuadratic,
    InteriorLog,
    InteriorInverseBarrier,
}

pub struct PointChoosen {}
pub struct SearchLineFound {}

#[derive(Clone)]
pub struct DescentController {
    pub previous_coordinates: Option<PosNDof<f64>>,
    pub current_coordinates: PosNDof<f64>,
    pub previous_gradient: Option<SpatialVectorNDof<f64, IsColl>>,
    pub current_gradient: SpatialVectorNDof<f64, IsColl>,
    pub previous_search_line: Option<SpatialVectorWithBasePointNDof<f64, IsColl>>,
    pub current_search_line: Option<SpatialVectorWithBasePointNDof<f64, IsColl>>,
    pub current_cost: Option<f64>,
    pub previous_hessian_approx: Option<Matrix<f64>>,
    pub current_hessian_approx: Option<Matrix<f64>>,
    pub optimization_settings: OptimizationSettings,
    pub line_searcher: LineSearcher,
    pub descent_logger: DescentLogger,
}

#[derive(Clone)]
pub struct DescentLogger {
    pub coordinates_history: Vec<Vec<f64>>, 
    pub convergence_history: Vec<f64>
}

impl Default for DescentLogger {
    fn default() -> Self {
        Self {
            coordinates_history: Vec::new(),
            convergence_history: Vec::new(),
        }
    }
}

impl DescentLogger {
    pub fn new(n_dof: usize) -> Self {
        let mut empty_logger = Self::default();
        for _ in 0..n_dof {
            empty_logger.coordinates_history.push(Vec::new());
        }
        empty_logger
    }
}

impl DescentController {
    // constructor functions
    // always sets the first search line (un bounded) with steepest descent
    pub fn new(
            cost_function: CostFunction, 
            cost_function_normalization_factor: Vec<f64>,
            cost_function_algebraic_derivative_supplied: bool,
            cost_function_algebraic_derivative: Option<AlgebraicDerivative>,
            constraint_equality_functions: Vec<CostFunction>, 
            constraint_inequality_functions: Vec<InequalityConstraint>, 
            search_point: PosNDof<f64>,
            optimization_settings: OptimizationSettings
        ) -> DescentController {

        let n_dof = search_point.n_dof();

        let mut cost_controller = CostController::new(
            cost_function,
            cost_function_normalization_factor,
            cost_function_algebraic_derivative_supplied,
            cost_function_algebraic_derivative,
            constraint_equality_functions,
            constraint_inequality_functions,
            optimization_settings.constraint_rules.clone(),
        );

        let gradient_placeholder = SpatialVectorNDof::new_from_direction(vec![0.; n_dof]);

        // let current_gradient = Self::uninit_current_slope(&mut cost_controller, &search_point, &optimization_settings);

        // let search_direction = -1. * &current_gradient;
        let search_line = SpatialVectorWithBasePointNDof {
            point: search_point.clone().x,
            vector: gradient_placeholder.vector.clone(),
            _orientation: PhantomData::<IsColl>,
        };

        let line_searcher = LineSearcher::new( search_line.clone(), cost_controller, optimization_settings.clone());


        let descent_logger = DescentLogger::new(n_dof);

        let mut descent_controller = DescentController {
            previous_coordinates: Option::None,
            current_coordinates: search_point.clone(),
            previous_gradient: Option::None,
            current_gradient: gradient_placeholder,
            previous_search_line: Option::None,
            current_search_line: Option::Some(search_line),
            current_cost: Option::None,
            previous_hessian_approx: Option::None,
            current_hessian_approx: Option::Some(Matrix::new_square_eye(n_dof, 1.)),
            optimization_settings,
            line_searcher,
            descent_logger,
        };

        descent_controller.set_local_gradient();
        descent_controller.current_search_line = Option::Some(SpatialVectorWithBasePointNDof {
            point: search_point.clone().x,
            vector: (-1. * descent_controller.current_gradient.clone()).vector,
            _orientation: PhantomData::<IsColl>,
        });
        descent_controller.line_searcher.search_line = descent_controller.current_search_line.clone().unwrap();

        descent_controller

    }

    // fn uninit_current_slope(cost_controller: &mut CostController, search_point: &PosNDof<f64>, optimization_settings: &OptimizationSettings) -> SpatialVectorNDof<f64, IsColl> {

    //     let n_degrees_of_freedom = search_point.n_dof();
    //     let derivative_per_dof: Collumn<f64>;
    //     let step_size = optimization_settings.differentiating_step_size;

    //     let mut objective_gradients = Collumn::new_with_constant_values(n_degrees_of_freedom, 0.);
    //     let mut constraint_gradient_row = Vec::<Row<f64>>::with_capacity(n_degrees_of_freedom);
    //     let penalty_function_derivatives = cost_controller.penalty_functions_derivatives_separated(&search_point.x);

    //     match optimization_settings.differentiation_rule {
    //         DifferentiationRule::CentralDifference => {

    //             for i_dof in 0.. n_degrees_of_freedom {
    //                 let mut position = search_point.clone();
    //                 position[i_dof] += step_size;
    //                 let cost_objective_function_at_small_step_forward = cost_controller.cost_at_x_exclusive_objective_function(&position.x);
    //                 position[i_dof] += -2. * step_size;
    //                 let cost_objective_function_at_small_step_bacward = cost_controller.cost_at_x_exclusive_objective_function(&position.x);
    //                 let cost_difference_objective_function = cost_objective_function_at_small_step_forward - cost_objective_function_at_small_step_bacward;
    //                 let gradient_objective_function = cost_difference_objective_function / (2. * step_size);
    //                 objective_gradients[i_dof] = gradient_objective_function;

    //                 let mut position = search_point.clone();
    //                 position[i_dof] += step_size;
    //                 let cost_contraint_functions_at_small_step_forward = cost_controller.constraint_values_at_x_no_rules_individual(&position.x);
    //                 position[i_dof] += -2. * step_size;
    //                 let cost_contraint_functions_at_small_step_bacward = cost_controller.constraint_values_at_x_no_rules_individual(&position.x);
    //                 let cost_difference_contraint_functions = cost_contraint_functions_at_small_step_forward - cost_contraint_functions_at_small_step_bacward;
    //                 let gradient_contraint_functions = cost_difference_contraint_functions / (2. * step_size);
    //                 constraint_gradient_row.push(gradient_contraint_functions)
    //                 // derivative_per_dof.push();
    //             }
    //         },
    //         DifferentiationRule::OneSided => {
    //             let cost_current_position = cost_controller.cost_at_pos(search_point);
                
    //             for i_dof in 0.. n_degrees_of_freedom {
    //                 let mut position = search_point.clone();
    //                 position[i_dof] += step_size;

    //                 let cost_objective_function_at_small_step_forward = cost_controller.cost_at_x_exclusive_objective_function(&position.x);
    //                 let cost_difference_objective_function = cost_objective_function_at_small_step_forward - cost_current_position;
    //                 let gradient_objective_function = cost_difference_objective_function / step_size;
    //                 objective_gradients[i_dof] = gradient_objective_function;

    //                 let mut cost_contraint_functions_at_small_step_forward = cost_controller.constraint_values_at_x_no_rules_individual(&position.x);
    //                 let mut cost_difference_contraint_functions = cost_contraint_functions_at_small_step_forward.subtract_scalar(cost_current_position).clone();
    //                 let gradient_contraint_functions = cost_difference_contraint_functions.divide_all_elements_by(step_size).clone();
    //                 constraint_gradient_row.push(gradient_contraint_functions)
    //             }
    //         }
    //     }
        
    //     let constraint_gradients = Matrix {
    //         rows: constraint_gradient_row
    //     };

    //     match optimization_settings.constraint_rules.constrained {
    //         OptimizationType::Constrained => {
    //             derivative_per_dof = objective_gradients;
    //         },
    //         OptimizationType::Unconstrained => {
    //             derivative_per_dof = objective_gradients + constraint_gradients * penalty_function_derivatives;
    //         },
    //     }

    //     SpatialVectorNDof::new_from_direction(derivative_per_dof.cells)


    //     // let number_dof = search_point.n_dof();
    //     // let differentiation_step_size = optimization_settings.differentiating_step_size;

    //     // let mut gradients = Vec::with_capacity(number_dof);
    //     // match optimization_settings.differentiation_rule {
    //     //     DifferentiationRule::CentralDifference => {
    //     //         for dof in 0..number_dof {
    //     //             let mut position = search_point.clone();
    //     //             position[dof] += differentiation_step_size;
    //     //             let function_value_at_small_step_forward = cost_controller.cost_at_pos(&position);
    //     //             position[dof] += - 2.*differentiation_step_size;
    //     //             let function_value_at_small_step_backwards = cost_controller.cost_at_pos(&position);
    //     //             let gradient = (function_value_at_small_step_forward - function_value_at_small_step_backwards) / (2. * differentiation_step_size);
    //     //             gradients.push(gradient)
    //     //         }
    //     //     },
    //     //     DifferentiationRule::OneSided => {
    //     //         let cost_current_position = cost_controller.cost_at_pos(search_point);
    //     //         for i_dof in 0 .. number_dof {
    //     //             let mut position = search_point.clone();
    //     //             position[i_dof] += differentiation_step_size;
    //     //             let function_value_at_small_step_forward = cost_controller.cost_at_pos(&position);
    //     //             let cost_difference = function_value_at_small_step_forward - cost_current_position;
    //     //             gradients.push(cost_difference / (differentiation_step_size));
    //     //         }
    //     //     },
    //     // }

    //     // SpatialVectorNDof::new_from_direction(gradients)
    // }

}

impl DescentController {
    // main iterator loop

    pub fn run_optimization(&mut self) {

        self.run_loggers();

        self.bracket_line_minumum();
        let mut new_search_point = self.locate_minumum_on_line();
        self.update_search_point_and_local_gradient(new_search_point);
        
        let mut index: usize = 0;
        while self.optimization_is_complete().not() {
            
            self.run_loggers();

            let new_search_line = self.find_descent_direction();

            self.line_searcher.update_line_searcher(new_search_line);

            self.bracket_line_minumum();

            new_search_point = self.locate_minumum_on_line();

            self.update_search_point_and_local_gradient(new_search_point);


            if self.maximum_itterations_reached(index) {
                break
            } else {
                index += 1;
            }
        }

        self.run_loggers();
    }


    fn find_descent_direction(&mut self) -> SpatialVectorWithBasePointNDof<f64, IsColl> {
        match self.optimization_settings.descent_method.clone() {
            DescentMethods::SteepestDescent => {
                self.use_steepest_descent();
            },
            DescentMethods::ConjugateDescent(conjugate_method) => {
                self.use_conjugate_descent(conjugate_method);
            },
            DescentMethods::QuasiNewton(quasi_newton_method) => {
                self.use_quasi_newton_method(quasi_newton_method);
            },
        }

        self.search_line()
    }

    fn bracket_line_minumum(&mut self) {
        match self.optimization_settings.line_search_extrapolation_method.clone() {
            LineSearchExtrapolationMethods::GoldenRuleExtrapolation => {
                self.line_searcher.find_bracket_golden_rule_extrapolation();
            },
        }
    }

    fn locate_minumum_on_line(&mut self) -> PosNDof<f64> {
        match self.optimization_settings.line_search_interpolation_method {
            LineSearchInterpolationMethods::GoldenRuleInterpolation => {
                self.line_searcher.golden_section_interpolation_iterate_untill_convergence();
            },
            LineSearchInterpolationMethods::QuadraticInterpolation => {
                self.line_searcher.quadratic_interpolation_iterate_untill_convergence();
            },
        }

        self.line_searcher.bracket_averaged_variables()
    }

    fn optimization_is_complete(&self) -> bool {
        self.local_gradient().length() < self.optimization_settings.optimization_end_threshold
    }

    fn maximum_itterations_reached(&self, index: usize) -> bool {
        match self.optimization_settings.max_itterations_optimization.clone() {
            Some(max_iterations) => {
                index == max_iterations
            },
            None => {
                false
            },
        }
    }
}

impl DescentController {
    // auxilary functions

    pub fn n_dof(&self) -> usize {
        self.current_coordinates.n_dof()
    }

    fn iterate_values(&mut self) -> &mut Self {
        // not supposed to be called by user

        self.previous_coordinates = Option::Some(self.current_coordinates.clone());

        self.previous_gradient = Option::Some(self.current_gradient.clone());

        if let Option::Some(_) = self.current_search_line {
            self.previous_search_line = self.current_search_line.clone();
            self.current_search_line = Option::None;
        }

        self.current_cost = Option::None;

        if let Option::Some(_) = self.current_hessian_approx {
            self.previous_hessian_approx = self.current_hessian_approx.clone();
            self.current_hessian_approx = Option::None;
        }

        self.line_searcher.reset_line_searcher();

        self
    }

    pub fn update_search_point_and_local_gradient(&mut self, new_search_point: PosNDof<f64>) -> &mut Self {
        self.iterate_values();
        self.current_coordinates = new_search_point;
        self.set_local_gradient();
        self
    }

    fn set_local_gradient(&mut self) ->  &mut Self {

        let n_degrees_of_freedom = self.current_coordinates.n_dof();
        let derivative_per_dof: Collumn<f64>;
        let search_vector = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &vec![0.; n_degrees_of_freedom]);
        let step_size = self.optimization_settings.differentiating_step_size;

        let mut objective_gradients = Collumn::new_with_constant_values(self.n_dof(), 0.);
        let mut constraint_gradient_row = Vec::<Row<f64>>::with_capacity(self.n_dof());
        let penalty_function_derivatives: Collumn<f64>;
        match self.optimization_settings.constraint_rules.constrained {
            OptimizationType::Constrained => {
                penalty_function_derivatives = self.penalty_functions_derivatives_separated(&search_vector.base_point_vec());
            },
            OptimizationType::Unconstrained => {
                penalty_function_derivatives = Collumn::new_form_vec(Vec::new());
            },
        }
        

        match self.optimization_settings.differentiation_rule {
            DifferentiationRule::CentralDifference => {

                for i_dof in 0.. n_degrees_of_freedom {
                    let cost_objective_function_at_small_step_forward = self.cost_at_x_exclusive_objective_function(&search_vector.volatile().set_vector_i_value(i_dof, step_size).end_position_vec());
                    let cost_objective_function_at_small_step_bacward = self.cost_at_x_exclusive_objective_function(&search_vector.volatile().set_vector_i_value(i_dof, -step_size).end_position_vec());
                    let cost_difference_objective_function = cost_objective_function_at_small_step_forward - cost_objective_function_at_small_step_bacward;
                    let gradient_objective_function = cost_difference_objective_function / (2. * step_size);
                    objective_gradients[i_dof] = gradient_objective_function;

                    if let OptimizationType::Constrained = self.optimization_settings.constraint_rules.constrained {
                        let cost_contraint_functions_at_small_step_forward = self.constraint_values_at_x_no_rules_individual(&search_vector.volatile().set_vector_i_value(i_dof, step_size).end_position_vec());
                        let cost_contraint_functions_at_small_step_bacward = self.constraint_values_at_x_no_rules_individual(&search_vector.volatile().set_vector_i_value(i_dof, -step_size).end_position_vec());
                        let cost_difference_contraint_functions = cost_contraint_functions_at_small_step_forward - cost_contraint_functions_at_small_step_bacward;
                        let gradient_contraint_functions = cost_difference_contraint_functions / (2. * step_size);
                        constraint_gradient_row.push(gradient_contraint_functions);
                    }
                    // derivative_per_dof.push();
                }
            },
            DifferentiationRule::OneSided => {
                let cost_current_position;
                if let Option::Some(current_cost) = self.current_cost {
                    cost_current_position = current_cost;
                } else {
                    cost_current_position = self.cost_at_current();
                }

                for i_dof in 0.. n_degrees_of_freedom {

                    let cost_objective_function_at_small_step_forward = self.cost_at_x_exclusive_objective_function(&search_vector.volatile().set_vector_i_value(i_dof, step_size).end_position_vec());
                    let cost_difference_objective_function = cost_objective_function_at_small_step_forward - cost_current_position;
                    let gradient_objective_function = cost_difference_objective_function / step_size;
                    objective_gradients[i_dof] = gradient_objective_function;

                    if let OptimizationType::Constrained = self.optimization_settings.constraint_rules.constrained {
                        let mut cost_contraint_functions_at_small_step_forward = self.constraint_values_at_x_no_rules_individual(&search_vector.volatile().set_vector_i_value(i_dof, step_size).end_position_vec());
                        let mut cost_difference_contraint_functions = cost_contraint_functions_at_small_step_forward.subtract_scalar(cost_current_position).clone();
                        let gradient_contraint_functions = cost_difference_contraint_functions.divide_all_elements_by(step_size).clone();
                        constraint_gradient_row.push(gradient_contraint_functions);
                    }
                }
            }
        }
        
        let constraint_gradients = Matrix {
            rows: constraint_gradient_row
        };

        match self.optimization_settings.constraint_rules.constrained {
            OptimizationType::Unconstrained => {
                derivative_per_dof = objective_gradients;
            },
            OptimizationType::Constrained => {
                derivative_per_dof = objective_gradients + constraint_gradients * penalty_function_derivatives;
            },
        }

        self.current_gradient = SpatialVectorNDof::new_from_direction(derivative_per_dof.cells);
        self
    }

    pub fn local_gradient(&self) -> SpatialVectorNDof<f64, IsColl> {
        self.current_gradient.clone()
    }

    pub fn search_line(&self) -> SpatialVectorWithBasePointNDof<f64, IsColl> {
        match self.current_search_line.clone() {
            Option::Some(search_line) => {
                search_line
            },
            Option::None => {
                panic!("There is no search line yet!");
            }
        }
    }

    fn cost_at_x(&mut self, x: &Vec<f64>) -> f64 {
        self.line_searcher.cost_controller.cost_at_x(x)
    }

    fn cost_at_x_exclusive_objective_function(&mut self, x: &Vec<f64>) -> f64 {
        self.line_searcher.cost_controller.cost_at_x_exclusive_objective_function(x)
    }

    fn constraint_values_at_x_no_rules_individual(&mut self, x: &Vec<f64>) -> Row<f64> {
        self.line_searcher.cost_controller.constraint_values_at_x_no_rules_individual(x)
    }

    fn penalty_functions_derivatives_separated(&mut self, x: &Vec<f64>) -> Collumn<f64> {
        self.line_searcher.cost_controller.penalty_functions_derivatives_separated(x)
    }

    fn cost_at_pos(&mut self, pos: &PosNDof<f64>) -> f64 {
        self.line_searcher.cost_controller.cost_at_pos(pos)
    }

    fn cost_at_current(&mut self) -> f64 {
        let current_cost = self.cost_at_pos(&self.current_coordinates.clone());
        self.current_cost = Option::Some(current_cost);
        current_cost
    }

    fn run_loggers(&mut self) {
        self.coordinates_logger();
        self.convergence_logger();
    }

    fn coordinates_logger(&mut self) {
        if self.optimization_settings.logging_settings.track_variable_history {
            for (index, coordinate) in self.current_coordinates.x.iter().enumerate() {
                self.descent_logger.coordinates_history[index].push(*coordinate);
            }
        }
    }

    fn convergence_logger(&mut self) {
        if self.optimization_settings.logging_settings.track_convergence_history {
            self.descent_logger.convergence_history.push(self.local_gradient().length())
        }
    }
}
