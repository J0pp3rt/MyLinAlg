use crate::{*};
#[derive(Debug, Clone, Copy)]
pub enum DescentMethods {
    SteepestDescent,
    ConjugateDescent(ConjugateGradient),
    QuasiNewton(QuasiNewton),
    QuadraticProgramming(QuasiNewton),
    NonLinearProgramming(QuasiNewton),
}
#[derive(Debug, Clone, Copy)]
pub enum ConjugateGradient {
    FletcherReeves,
    PolakRibiere
}
#[derive(Debug, Clone, Copy)]
pub enum QuasiNewton {
    BFGS,
    BFGSNoInverse,
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
    pub coordinates_upper_range: Option<PosNDof<f64>>,
    pub coordinates_lower_range: Option<PosNDof<f64>>,
    pub first_gradient_length: f64,
    pub previous_gradient: Option<SpatialVectorNDof<f64, IsColl>>,
    pub current_gradient: SpatialVectorNDof<f64, IsColl>,
    pub previous_constraint_gradient: Option<Vec<SpatialVectorNDof<f64, IsColl>>>,
    pub current_constraint_gradient: Option<Vec<SpatialVectorNDof<f64, IsColl>>>,
    pub previous_search_line: Option<SpatialVectorWithBasePointNDof<f64, IsColl>>,
    pub current_search_line: Option<SpatialVectorWithBasePointNDof<f64, IsColl>>,
    pub current_cost: Option<f64>,
    pub previous_hessian_approx: Option<Matrix<f64>>,
    pub current_hessian_approx: Option<Matrix<f64>>,
    pub previous_constraint_hessian_approx: Option<Vec<Matrix<f64>>>,
    pub current_constraint_hessian_approx: Option<Vec<Matrix<f64>>>,
    pub optimization_settings: OptimizationSettings,
    pub line_searcher: LineSearcher,
    pub descent_logger: DescentLogger,
    pub non_linear_logger: NonLinearLogger,
}

#[derive(Clone)]
pub struct NonLinearLogger {
    pub is_first_cycle: bool,
    pub active_hessian: Matrix<f64>,
    pub secondary_hessian: Matrix<f64>,
    pub cycles_per_active_hessian: isize,
    pub cycles_left_on_active_before_switch: isize,
    pub cycles_needed_for_building: isize,
    pub cycles_left_before_building_secondary: isize,
    pub secondary_building: bool,
    pub end_condition: f64,
    pub end_condition_normalization_factor: f64,
}

#[derive(Clone)]
pub struct DescentLogger {
    pub coordinates_history: Vec<Vec<f64>>, 
    pub convergence_history: Vec<f64>,
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
            coordinates_upper_bound:Option<PosNDof<f64>>,
            coordinates_lower_bound:Option<PosNDof<f64>>, 
            optimization_settings: OptimizationSettings
        ) -> DescentController {

        let n_dof = search_point.n_dof();

        let test_run_objective_function = (cost_function)(&search_point.x);
        let n_constraints_in_objective_function = test_run_objective_function.model_constraints.len();

        let mut cost_controller = CostController::new(
            cost_function,
            cost_function_normalization_factor,
            cost_function_algebraic_derivative_supplied,
            cost_function_algebraic_derivative,
            constraint_equality_functions,
            constraint_inequality_functions,
            optimization_settings.constraint_rules.clone(),
            optimization_settings,
            n_constraints_in_objective_function
        );

        let gradient_placeholder = SpatialVectorNDof::new_from_direction(vec![0.; n_dof]);

        // let current_gradient = Self::uninit_current_slope(&mut cost_controller, &search_point, &optimization_settings);

        // let search_direction = -1. * &current_gradient;
        let search_line = SpatialVectorWithBasePointNDof {
            point: search_point.clone().x,
            vector: gradient_placeholder.vector.clone(),
            _orientation: PhantomData::<IsColl>,
        };

        let n_constraint = cost_controller.n_constraints();

        let line_searcher = LineSearcher::new( search_line.clone(), cost_controller, optimization_settings.clone());


        let descent_logger = DescentLogger::new(n_dof);

        let hessian_matrix: Option<Matrix<f64>>;
        let hessian_matrix_contraint: Option<Vec<Matrix<f64>>>;
        match optimization_settings.descent_method {
            DescentMethods::SteepestDescent => {
                hessian_matrix = Option::None;
                hessian_matrix_contraint = Option::None;
            },
            DescentMethods::ConjugateDescent(_) => {
                hessian_matrix = Option::None;
                hessian_matrix_contraint = Option::None;
            },
            DescentMethods::QuasiNewton(_) => {
                hessian_matrix = Option::Some(Matrix::new_square_eye(n_dof, 1.));
                hessian_matrix_contraint = Option::None;
            },
            DescentMethods::QuadraticProgramming(_) => {
                hessian_matrix = Option::Some(Matrix::new_square_eye(n_dof, 1.));
                hessian_matrix_contraint = Option::None;
            },
            DescentMethods::NonLinearProgramming(_) => {
                hessian_matrix = Option::Some(Matrix::new_square_eye(n_dof, 1.));
                let mut constraint_hessian_vec = Vec::<Matrix<f64>>::new();
                let hessian_matrix_contraint_base = Matrix::new_square_eye(n_dof, 1.);
                for _ in 0.. n_constraint {
                    constraint_hessian_vec.push(hessian_matrix_contraint_base.clone());
                }
                hessian_matrix_contraint = Option::Some(constraint_hessian_vec);
            }
        }
        // Option::Some(Matrix::new_square_eye(n_dof, 1.))

        let non_linear_logger = NonLinearLogger {
            is_first_cycle: true,
            active_hessian: Matrix::new_square_eye(n_dof, 1.),
            secondary_hessian: Matrix::new_square_eye(n_dof, 1.),
            cycles_per_active_hessian: 10,
            cycles_left_on_active_before_switch: 10,
            cycles_needed_for_building: 5,
            cycles_left_before_building_secondary: 5,
            secondary_building: false,
            end_condition: 1.,
            end_condition_normalization_factor: 1.,
        };

        let mut descent_controller = DescentController {
            previous_coordinates: Option::None,
            current_coordinates: search_point.clone(),
            coordinates_upper_range: coordinates_upper_bound,
            coordinates_lower_range: coordinates_lower_bound,
            first_gradient_length: 1.,
            previous_gradient: Option::None,
            current_gradient: gradient_placeholder,
            previous_constraint_gradient: Option::None,
            current_constraint_gradient: Option::None,
            previous_search_line: Option::None,
            current_search_line: Option::Some(search_line),
            current_cost: Option::None,
            previous_hessian_approx: Option::None,
            current_hessian_approx: hessian_matrix,
            previous_constraint_hessian_approx: Option::None,
            current_constraint_hessian_approx: hessian_matrix_contraint,
            optimization_settings,
            line_searcher,
            descent_logger,
            non_linear_logger,

        };

        descent_controller.set_local_gradient();
        // descent_controller.first_gradient_length = 1.;
        descent_controller.first_gradient_length = descent_controller.current_gradient.length();
        // descent_controller.current_gradient.normalize().scale(descent_controller.first_gradient_length);
        // descent_controller.current_gradient.normalize().scale(0.001);
        descent_controller.current_search_line = Option::Some(SpatialVectorWithBasePointNDof {
            point: search_point.clone().x,
            vector: (-1. * descent_controller.current_gradient.clone()).vector,
            _orientation: PhantomData::<IsColl>,
        });
        descent_controller.line_searcher.search_line = descent_controller.current_search_line.clone().unwrap();

        match descent_controller.optimization_settings.descent_method {
            DescentMethods::NonLinearProgramming(_) => {
                descent_controller.non_linear_logger.end_condition_normalization_factor = descent_controller.non_linear_derivative();
                descent_controller.non_linear_logger.end_condition = descent_controller.non_linear_logger.end_condition / descent_controller.non_linear_logger.end_condition_normalization_factor;
            },
            _ => {}
        }

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

        self.bracket_line_minumum(true);
        let mut new_search_point = self.locate_minumum_on_line();
        self.update_search_point_and_local_gradient(new_search_point);
        
        let mut index: usize = 0;
        while self.optimization_is_complete().not() {
            
            // println!("~~~~~~~~~~~~~~~~~~~~~~~");
            // println!("new itteration, current location: {:?}", self.line_searcher.search_line.base_point_vec());
            // println!("~~~~~~~~~~~~~~~~~~~~~~~");
            self.run_loggers();
            let new_search_line = self.find_descent_direction();


            self.line_searcher.update_line_searcher(new_search_line);


            self.bracket_line_minumum(false);

            new_search_point = self.locate_minumum_on_line();

            self.update_search_point_and_local_gradient(new_search_point);

            if self.maximum_itterations_reached(index) {
                break
            } else {
                index += 1;
            }
        }

        self.run_loggers();
        println!("done!");
        println!("iteration: {:}, current_position: {:?}, function_evals: {}", index, self.current_coordinates.x, self.line_searcher.cost_controller.number_cost_function_evaluation);
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
                self.use_quasi_newton_method(quasi_newton_method, true);
            },
            DescentMethods::QuadraticProgramming(quasi_newton_method) => {
                self.use_quadratic_programming_method(quasi_newton_method);
            },
            DescentMethods::NonLinearProgramming(quasi_newton_method) => {
                self.use_non_linear_programming_method(quasi_newton_method);
            }
        }

        self.search_line()
    }

    fn bracket_line_minumum(&mut self, is_first_step: bool) {
        match self.optimization_settings.line_search_extrapolation_method.clone() {
            LineSearchExtrapolationMethods::GoldenRuleExtrapolation => {
                self.line_searcher.find_bracket_golden_rule_extrapolation();
            },
            LineSearchExtrapolationMethods::NoLineSearch => {
                if is_first_step {
                    self.line_searcher.search_line.normalize().move_base_to_end_position();
                    // self.line_searcher.search_line.move_base_to_end_position();
                } else {
                    // let mut  calculated_new_point = PosNDof {
                    //     x: self.line_searcher.search_line.volatile().move_base_to_end_position().vector.clone()
                    // };

                    // let check_range: bool;
                    // match (&self.coordinates_lower_range, &self.coordinates_upper_range) {
                    //     (Some(_), Some(_)) => {check_range = true},
                    //     _ => {check_range = false}
                    // }

                    // enum OutOfRange {
                    //     TooHigh(usize),
                    //     TooLow(usize),
                    // }
    
                    // if check_range {
                    //     let minimum_scaling = 0.01;

                    //     let upper_bound = self.coordinates_upper_range.clone().unwrap().x();
                    //     let lower_bound = self.coordinates_lower_range.clone().unwrap().x();
    
                    //     let mut max_scaling_factor = 1.;
        
                    //     for (i_dof, coordinate) in calculated_new_point.x.iter().enumerate() {
                    //         let upper_bound_coordinate = upper_bound[i_dof];
                    //         let lower_bound_coordinate = lower_bound[i_dof];

                    //         if *coordinate > upper_bound_coordinate {
                    //             let coordinate_max_scaling = 
                    //                 self.line_searcher.search_line.scaling_factor_untill_coordinate(upper_bound_coordinate, i_dof);

                    //             match coordinate_max_scaling {
                    //                 MaxScalingFactorReturn::ScalingFactor(scaling_factor) => {
                    //                     if scaling_factor > max_scaling_factor {
                    //                         max_scaling_factor = scaling_factor;
                    //                     } 
                    //                 },
                    //                 MaxScalingFactorReturn::ZeroDistance => {
                    //                     max_scaling_factor = minimum_scaling;
                    //                 },
                    //                 MaxScalingFactorReturn::NotBound => {
                    //                     max_scaling_factor = minimum_scaling;
                    //                 },
                    //             }

                    //         } else if *coordinate < lower_bound_coordinate {
                    //             let coordinate_max_scaling = 
                    //                 self.line_searcher.search_line.scaling_factor_untill_coordinate(lower_bound_coordinate, i_dof);

                    //             match coordinate_max_scaling {
                    //                 MaxScalingFactorReturn::ScalingFactor(scaling_factor) => {
                    //                     if scaling_factor < max_scaling_factor {
                    //                         max_scaling_factor = scaling_factor;
                    //                     } 
                    //                 },
                    //                 MaxScalingFactorReturn::ZeroDistance => {
                    //                     max_scaling_factor = minimum_scaling;
                    //                 },
                    //                 MaxScalingFactorReturn::NotBound => {
                    //                     max_scaling_factor = minimum_scaling;
                    //                 },
                    //             }
                    //         }
                    //     }

                    //     self.line_searcher.search_line.scale(max_scaling_factor);

                    // }
    
                    // calculated_new_point
                    self.line_searcher.search_line.move_base_to_end_position();
    
                }
            },
        }
    }

    fn locate_minumum_on_line(&mut self) -> PosNDof<f64> {
        match self.optimization_settings.line_search_interpolation_method {
            LineSearchInterpolationMethods::GoldenRuleInterpolation => {
                self.line_searcher.golden_section_interpolation_iterate_untill_convergence();
                self.line_searcher.bracket_averaged_variables()
            },
            LineSearchInterpolationMethods::QuadraticInterpolation => {
                self.line_searcher.quadratic_interpolation_iterate_untill_convergence();
                self.line_searcher.bracket_averaged_variables()
            },
            LineSearchInterpolationMethods::NoLineSearch => {
                PosNDof {
                    x: self.line_searcher.coordinates_of_lower_bound()
                }
        }
    }
    }

    fn optimization_is_complete(&mut self) -> bool {

        let optimization_end_threshold = self.optimization_threshold_corrected();
        let local_gradient_normalized = self.optimization_threshold_test_value();

        let constraints_met: bool;
        match self.optimization_settings.constraint_rules.constrained {
            OptimizationType::ConstrainedProgramming => {
                let constraints_are_met = self.line_searcher.cost_controller.constraints_are_met(&self.current_coordinates.x, 0.001);

                constraints_met = constraints_are_met;
                // constraints_met = true; // otherwise never accepted.
                if constraints_are_met {
                    println!("local_gradient: {}, constraints_met {}, threshold: {}", local_gradient_normalized, constraints_are_met, optimization_end_threshold);
                }
                
            },
            OptimizationType::ConstrainedMerit => {
                constraints_met = true;
            },
            OptimizationType::Unconstrained => {
                constraints_met = true;
            },
        }

        let completion_condition = local_gradient_normalized < optimization_end_threshold && constraints_met;

        match self.optimization_settings.descent_method {
            DescentMethods::NonLinearProgramming(_) => {
                if constraints_met && (local_gradient_normalized < optimization_end_threshold).not() && self.line_searcher.cost_controller.n_constraints() > 0 {
                    self.optimization_settings.optimization_end_threshold = self.optimization_settings.optimization_end_threshold * 1.0;
                }
            },
            _ => {},
        }

        completion_condition
    }

    fn optimization_threshold_corrected(&self) -> f64 {
        let optimization_end_threshold: f64;
        match self.line_searcher.cost_controller.constraint_rules.constrained {
            OptimizationType::ConstrainedMerit => {
                // optimization_end_threshold = self.optimization_settings.optimization_end_threshold * (self.line_searcher.cost_controller.constraint_rules.magnification_factor / 2.);
                optimization_end_threshold = self.optimization_settings.optimization_end_threshold;
            },
            OptimizationType::ConstrainedProgramming => {
                optimization_end_threshold = self.optimization_settings.optimization_end_threshold;
            }
            OptimizationType::Unconstrained => {
                optimization_end_threshold = self.optimization_settings.optimization_end_threshold;
            },
        }
        optimization_end_threshold
    }

    fn optimization_threshold_test_value(&mut self) -> f64 {
        match self.optimization_settings.descent_method {
            DescentMethods::NonLinearProgramming(_) => {
                self.non_linear_derivative()
            },
            _ => {
                self.current_gradient.length() / self.first_gradient_length
            }
        }
        
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

    pub fn gradient_length_ratio_last_iteration(&self) -> f64 {

        let length_ratio: f64;
        match self.previous_gradient.as_ref() {
            Some(previous_gradient) => {
                length_ratio = self.current_gradient.length() / previous_gradient.length();
            },
            None => {
                length_ratio = 1.;
            },
        }

        // println!("length_ratio = {}", length_ratio);
        length_ratio
    }

    fn iterate_values(&mut self) -> &mut Self {
        // not supposed to be called by user

        // let gradient_length_ratio_last_iteration = self.gradient_length_ratio_last_iteration();
        // match gradient_length_ratio_last_iteration <= 1. {
        //     true => {
        //         self.line_searcher.optimization_settings.line_search_threshold = self.line_searcher.optimization_settings.line_search_threshold * gradient_length_ratio_last_iteration;
        //     },
        //     _ => {}
        // }

        // self.line_searcher.cost_controller.itterate_constraint_rules();

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

        if let Option::Some(_) = self.current_constraint_hessian_approx {
            self.previous_constraint_hessian_approx = self.current_constraint_hessian_approx.clone();
            self.current_constraint_hessian_approx = Option::None;
        }

        if let Option::Some(_) = self.current_constraint_gradient {
            self.previous_constraint_gradient = self.current_constraint_gradient.clone();
            self.current_constraint_gradient = Option::None;
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
        let derivative_per_dof: SpatialVectorNDof<f64, IsColl>;
        let search_vector = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &vec![0.; n_degrees_of_freedom]);
        let step_size = self.optimization_settings.differentiating_step_size;

        let mut objective_gradients = self.line_searcher.cost_controller.derivative_objective_function(&self.current_coordinates.x);
        
        // let normalization_factor_objective_gradients = 1. / (self.first_gradient_length);

        // objective_gradients.normalize();

        // objective_gradients = objective_gradients * normalization_factor_objective_gradients ;


        let mut constraint_gradient_row = Vec::<Row<f64>>::with_capacity(self.n_dof());
        let penalty_function_derivatives: Collumn<f64>;
        match self.optimization_settings.constraint_rules.constrained {
            OptimizationType::ConstrainedMerit => {
                penalty_function_derivatives = self.penalty_functions_derivatives_separated(&search_vector.base_point_vec());
            },
            OptimizationType::ConstrainedProgramming => {
                penalty_function_derivatives = Collumn::new_form_vec(Vec::new());
            },
            OptimizationType::Unconstrained => {
                penalty_function_derivatives = Collumn::new_form_vec(Vec::new());
            },
        }
        

        match self.optimization_settings.differentiation_rule {
            DifferentiationRule::CentralDifference => {

                for i_dof in 0.. n_degrees_of_freedom {
                    if let OptimizationType::ConstrainedMerit = self.optimization_settings.constraint_rules.constrained {
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
                match self.current_cost {
                    Some(cost) => {
                        cost_current_position = cost;
                    },
                    None => {
                        cost_current_position = self.cost_at_current();
                    },
                }

                for i_dof in 0.. n_degrees_of_freedom {

                    if let OptimizationType::ConstrainedMerit = self.optimization_settings.constraint_rules.constrained {
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
            OptimizationType::ConstrainedMerit => {
                derivative_per_dof = objective_gradients + constraint_gradients * penalty_function_derivatives;
            },
            OptimizationType::ConstrainedProgramming => {
                derivative_per_dof = objective_gradients;
                self.current_constraint_gradient = Option::Some(self.line_searcher.cost_controller.derivative_all_inequality_constraints(&self.current_coordinates.x));
            },
        }

        // println!("found derivatives: {:?}", derivative_per_dof.vector);
        self.current_gradient = derivative_per_dof;
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
            match self.optimization_settings.descent_method {
                DescentMethods::NonLinearProgramming(_) => {
                    self.descent_logger.convergence_history.push(
                        self.non_linear_logger.end_condition
                    )
                },
                _ => {
                    let value = self.optimization_threshold_test_value();
                    self.descent_logger.convergence_history.push(
                        value
                    )
                }
            }

        }
    }
}
