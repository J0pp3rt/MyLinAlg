use crate::{*};
#[derive(Clone, Debug)]
pub enum DescentMethods {
    SteepestDescent,
    ConjugateDescent(ConjugateGradient),
    QuasiNewton(QuasiNewton)
}
#[derive(Clone, Debug)]
pub enum ConjugateGradient {
    FletcherReeves,
    PolakRibiere
}
#[derive(Clone, Debug)]
pub enum QuasiNewton {
    BFGS,
    BFGSBeforeInverse,
    BFGSBeforeInverseMoreStable,
    DFP,
    DFPMoreStable
}

#[derive(Clone, Debug)]
pub enum ConstraintMerit {
    ExteriorQuadratic,
    InteriorLog,
    InteriorInverseBarrier,
}

pub struct PointChoosen {}
pub struct SearchLineFound {}


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

        let mut cost_controller = CostController::new(
            cost_function,
            cost_function_normalization_factor,
            cost_function_algebraic_derivative_supplied,
            cost_function_algebraic_derivative,
            constraint_equality_functions,
            constraint_inequality_functions,
            optimization_settings.cost_rule.clone(),
        );

        let current_gradient = Self::uninit_current_slope(&mut cost_controller, &search_point, &optimization_settings);

        let search_direction = -1. * &current_gradient;
        let search_line = SpatialVectorWithBasePointNDof {
            point: search_point.clone().x,
            vector: search_direction.vector,
            _orientation: PhantomData::<IsColl>,
        };

        let line_searcher = LineSearcher::new( search_line.clone(), cost_controller, optimization_settings.clone());
        let n_dof = search_point.n_dof();

        DescentController {
            previous_coordinates: Option::None,
            current_coordinates: search_point.clone(),
            previous_gradient: Option::None,
            current_gradient,
            previous_search_line: Option::None,
            current_search_line: Option::Some(search_line),
            current_cost: Option::None,
            previous_hessian_approx: Option::None,
            current_hessian_approx: Option::Some(Matrix::new_square_eye(n_dof, 1.)),
            optimization_settings,
            line_searcher,
        }
    }

    fn uninit_current_slope(cost_controller: &mut CostController, search_point: &PosNDof<f64>, optimization_settings: &OptimizationSettings) -> SpatialVectorNDof<f64, IsColl> {
        let number_dof = search_point.n_dof();
        let differentiation_step_size = optimization_settings.differentiating_step_size;

        let mut gradients = Vec::with_capacity(number_dof);
        match optimization_settings.differentiation_rule {
            DifferentiationRule::CentralDifference => {
                for dof in 0..number_dof {
                    let mut position = search_point.clone();
                    position[dof] += differentiation_step_size;
                    let function_value_at_small_step_forward = cost_controller.cost_at_pos(&position);
                    position[dof] += - 2.*differentiation_step_size;
                    let function_value_at_small_step_backwards = cost_controller.cost_at_pos(&position);
                    let gradient = (function_value_at_small_step_forward - function_value_at_small_step_backwards) / (2. * differentiation_step_size);
                    gradients.push(gradient)
                }
            },
            DifferentiationRule::OneSided => {
                let cost_current_position = cost_controller.cost_at_pos(search_point);
                for i_dof in 0 .. number_dof {
                    let mut position = search_point.clone();
                    position[i_dof] += differentiation_step_size;
                    let function_value_at_small_step_forward = cost_controller.cost_at_pos(&position);
                    let cost_difference = function_value_at_small_step_forward - cost_current_position;
                    gradients.push(cost_difference / (differentiation_step_size));
                }
            },
        }

        SpatialVectorNDof::new_from_direction(gradients)
    }

}

impl DescentController {
    // main iterator loop

    pub fn run_optimization(&mut self) {

        self.bracket_line_minumum();
        let mut new_search_point = self.locate_minumum_on_line();
        self.update_search_point_and_local_gradient(new_search_point);
        
        let mut index: usize = 0;
        while self.optimization_is_complete().not() {
            
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
        let mut derivative_per_dof = Vec::<f64>::with_capacity(n_degrees_of_freedom);
        let search_vector = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &vec![0.; n_degrees_of_freedom]);
        let step_size = self.optimization_settings.differentiating_step_size;

        match self.optimization_settings.differentiation_rule {
            DifferentiationRule::CentralDifference => {
                for i_dof in 0.. n_degrees_of_freedom {
                    let cost_function_at_small_step_forward = self.cost_at_x(&search_vector.volatile().set_vector_i_value(i_dof, step_size).end_position_vec());
                    let cost_function_at_small_step_bacward = self.cost_at_x(&search_vector.volatile().set_vector_i_value(i_dof, -step_size).end_position_vec());
                    let cost_difference = cost_function_at_small_step_forward - cost_function_at_small_step_bacward;
                    derivative_per_dof.push(cost_difference / (2. * step_size));
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
                    let cost_function_at_small_step_forward = self.cost_at_x(&search_vector.volatile().set_vector_i_value(i_dof, step_size).end_position_vec());
                    let cost_difference = cost_function_at_small_step_forward - cost_current_position;
                    derivative_per_dof.push(cost_difference / (step_size));
                }
            },
        }

        self.current_gradient = SpatialVectorNDof::new_from_direction(derivative_per_dof);
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

    fn cost_at_pos(&mut self, pos: &PosNDof<f64>) -> f64 {
        self.line_searcher.cost_controller.cost_at_pos(pos)
    }

    fn cost_at_current(&mut self) -> f64 {
        let current_cost = self.cost_at_pos(&self.current_coordinates.clone());
        self.current_cost = Option::Some(current_cost);
        current_cost
    }
}
