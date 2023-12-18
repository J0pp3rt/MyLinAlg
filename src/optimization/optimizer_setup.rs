use crate::{*};

pub struct OptimizerSetup {
    cost_function: CostFunction,
    cost_function_normalization_factor: Vec<f64>,
    cost_function_algebraic_derivative_supplied: bool,
    cost_function_algebraic_derivative: Option<AlgebraicDerivative>,
    constaint_equality_functions: Vec<CostFunction>,
    constaint_inequality_functions: Vec<InequalityConstraint>,
    initial_coordinates: PosNDof<f64>,

    optimization_settings: OptimizationSettings,
    logging_settings: LoggingSettings,
}

impl Default for OptimizerSetup {
    fn default() -> Self {
        let default_cost_function: CostFunction = Box::new(|x: &Vec<f64>| get_cost_function_roosenbruck(x));
        let default_begin_variables = PosNDof{x: vec![-2., -3.]};

        let default_optimization_settings = OptimizationSettings::default();

        Self { 
            cost_function: default_cost_function, 
            cost_function_algebraic_derivative_supplied: false,
            cost_function_algebraic_derivative: Option::None,
            constaint_equality_functions: Vec::new(), 
            constaint_inequality_functions: Vec::new(), 
            cost_function_normalization_factor: Vec::new(),
            initial_coordinates: default_begin_variables,
            optimization_settings: default_optimization_settings,
            logging_settings: LoggingSettings::default(),
         }
    }
}



#[derive(Clone)]
pub struct OptimizationSettings {
    pub line_search_extrapolation_method: LineSearchExtrapolationMethods,
    pub line_search_interpolation_method: LineSearchInterpolationMethods,
    pub descent_method: DescentMethods,
    pub cost_rule: CostRule,
    pub differentiation_rule: DifferentiationRule,
    pub max_itterations_optimization: Option<usize>,
    pub max_itterations_linesearch: Option<usize>,
    pub optimization_end_threshold: f64,
    pub differentiating_step_size: f64,
    pub line_search_initial_step_size: f64,
    pub line_search_threshold: f64,
}

impl Default for OptimizationSettings {
    fn default() -> Self {

        let default_optimization_end_threshold = 0.5;
        let default_line_search_initial_step_size = 1.;
        let default_differentiating_step_size = 0.00000001;
        let default_line_search_threshold = 0.1;

        Self { 
            line_search_extrapolation_method: LineSearchExtrapolationMethods::GoldenRuleExtrapolation, 
            line_search_interpolation_method: LineSearchInterpolationMethods::GoldenRuleInterpolation, 
            descent_method: DescentMethods::SteepestDescent, 
            cost_rule: CostRule::Unconstrained,
            differentiation_rule: DifferentiationRule::CentralDifference,
            max_itterations_optimization: Option::None,
            max_itterations_linesearch: Option::None,
            optimization_end_threshold: default_optimization_end_threshold,
            differentiating_step_size: default_differentiating_step_size,
            line_search_initial_step_size: default_line_search_initial_step_size,
            line_search_threshold: default_line_search_threshold,
        }
    }
}


pub struct LoggingSettings {
    track_variable_history: bool,
    track_convergence_history: bool,
    iteration_print_variables: bool,
    iteration_print_convergence: bool,
    iteration_print_number: bool,
}

impl Default for LoggingSettings {
    fn default() -> Self {
        Self { 
            track_variable_history: false, 
            track_convergence_history: false, 
            iteration_print_variables: false, 
            iteration_print_convergence: false, 
            iteration_print_number: false 
        }
    }
}


impl OptimizerSetup {
    pub fn new() -> OptimizerSetup {
        Self::default()
    }

    pub fn cost_function(&mut self, cost_function: CostFunction) -> &mut Self {
        
        self.cost_function = cost_function;
        
        self
    }

    pub fn cost_function_derivative(&mut self, derivative: AlgebraicDerivative) -> &mut Self {
        self.cost_function_algebraic_derivative_supplied = true;
        self.cost_function_algebraic_derivative = Option::Some(derivative);

        self
    }

    pub fn add_equality_constraint_function(&mut self, equality_contraint: CostFunction) -> &mut Self {
        self.constaint_equality_functions.push(equality_contraint);

        self
    }

    pub fn add_inequality_constraint_function(&mut self, inequality_contraint: CostFunction) -> &mut Self {
        let constraint = InequalityConstraint {
            constraint_function: inequality_contraint,
            algebraic_derivative_supplied: false,
            algebraic_derivative: Option::None,
        };

        self.constaint_inequality_functions.push(constraint);

        self
    }

    pub fn add_inequality_constraint_function_with_derivative(&mut self, inequality_contraint: CostFunction, derivative: AlgebraicDerivative) -> &mut Self {
        let constraint = InequalityConstraint {
            constraint_function: inequality_contraint,
            algebraic_derivative_supplied: true,
            algebraic_derivative: Option::Some(derivative),
        };

        self.constaint_inequality_functions.push(constraint);

        self
    }

    pub fn cost_function_normalization_factor(&mut self, normalization_factors: Vec<f64>) -> &mut Self {
        self.cost_function_normalization_factor = normalization_factors;

        self
    }

    pub fn initial_values(&mut self, initial_coordinates: PosNDof<f64>) -> &mut Self {
        self.initial_coordinates = initial_coordinates;

        self
    }

    pub fn line_search_extrapolation_method(&mut self, extrapolation_method: LineSearchExtrapolationMethods) -> &mut Self {
        self.optimization_settings.line_search_extrapolation_method = extrapolation_method;

        self
    }

    pub fn line_search_interpolation_method(&mut self, interpolation_method: LineSearchInterpolationMethods) -> &mut Self {
        self.optimization_settings.line_search_interpolation_method = interpolation_method;

        self
    }

    pub fn descent_method(&mut self, descent_method: DescentMethods) -> &mut Self {
        self.optimization_settings.descent_method = descent_method;

        self
    }

    pub fn optimization_end_threshold(&mut self, optimization_threshold: f64) -> &mut Self {
        self.optimization_settings.optimization_end_threshold = optimization_threshold;

        self
    }


    pub fn differentiating_step_size(&mut self, differentiating_step_size: f64) -> &mut Self {
        self.optimization_settings.differentiating_step_size = differentiating_step_size;

        self
    }


    pub fn line_search_initial_step_size(&mut self, line_search_initial_step_size: f64) -> &mut Self {
        self.optimization_settings.line_search_initial_step_size = line_search_initial_step_size;

        self
    }


    pub fn line_search_threshold(&mut self, line_search_threshold: f64) -> &mut Self {
        self.optimization_settings.line_search_threshold = line_search_threshold;

        self
    }

    pub fn max_itterations_optimization(&mut self, max_iteration_opimization: usize) -> &mut Self {
        self.optimization_settings.max_itterations_optimization = Option::Some(max_iteration_opimization);

        self
    }


    pub fn max_itterations_linesearch(&mut self, max_itterations_linesearch: usize) -> &mut Self {
        self.optimization_settings.max_itterations_linesearch = Option::Some(max_itterations_linesearch);

        self
    }


    pub fn logging_track_variable_history(&mut self, track_variable_history: bool) -> &mut Self {
        self.logging_settings.track_variable_history = track_variable_history;

        self
    }

    pub fn logging_track_convergence_history(&mut self, track_convergence_history: bool) -> &mut Self {
        self.logging_settings.track_convergence_history = track_convergence_history;

        self
    }

    pub fn logging_iteration_print_variables(&mut self, iteration_print_variables: bool) -> &mut Self {
        self.logging_settings.iteration_print_variables = iteration_print_variables;

        self
    }

    pub fn logging_iteration_print_convergence(&mut self, iteration_print_convergence: bool) -> &mut Self {
        self.logging_settings.iteration_print_convergence = iteration_print_convergence;

        self
    }

    pub fn logging_iteration_print_number(&mut self, iteration_print_number: bool) -> &mut Self {
        self.logging_settings.iteration_print_number = iteration_print_number;

        self
    }

    pub fn activate(self) -> Optimizer {
        let descent_controller = DescentController::new(
            self.cost_function,
            self.cost_function_normalization_factor,
            self.cost_function_algebraic_derivative_supplied,
            self.cost_function_algebraic_derivative,
            self.constaint_equality_functions,
            self.constaint_inequality_functions,
            self.initial_coordinates,
            self.optimization_settings,
        );

        let default_empty_array_buffer: usize = 1000;

        Optimizer::new(
            descent_controller, 
            Vec::with_capacity(default_empty_array_buffer), 
            Vec::with_capacity(default_empty_array_buffer), 
        )
    }
}
