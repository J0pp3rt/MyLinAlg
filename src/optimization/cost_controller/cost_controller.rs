use crate::{*};

#[derive(Clone)]
pub struct CostController {
    pub cost_function: CostFunction,
    pub cost_function_normalization_factor: Vec<f64>,
    pub constraint_equality_functions: Vec<CostFunction>,
    pub constraint_inequality_functions: Vec<InequalityConstraint>,
    pub number_cost_function_evaluation: usize,
    pub number_constraint_function_evaluations: usize,
    pub constraint_rules: ConstrainedRules,
    pub optimization_settings: OptimizationSettings,
    pub n_constraints_objective_result: usize,
}

impl CostController {
    pub fn new(
        cost_function: CostFunction, 
        cost_function_normalization_factor: Vec<f64>,
        cost_function_algebraic_derivative_supplied: bool,
        cost_function_algebraic_derivative: Option<AlgebraicDerivative>,
        constraint_equality_functions: Vec<CostFunction>,
        constraint_inequality_functions: Vec<InequalityConstraint>,
        constraint_rules: ConstrainedRules,
        optimization_settings: OptimizationSettings,
        n_constraints_in_objective: usize,
    ) -> Self {

        Self { 
            cost_function, 
            cost_function_normalization_factor, 
            constraint_equality_functions, 
            constraint_inequality_functions,
            number_cost_function_evaluation: 0,
            number_constraint_function_evaluations: 0,
            constraint_rules,
            optimization_settings,
            n_constraints_objective_result: n_constraints_in_objective
        }
    }
}

impl CostController {
    pub fn n_constraints(&mut self) -> usize {
        self.constraint_inequality_functions.len() + self.n_constraints_objective_result
    }

    pub fn cost_at_x(&mut self, x: &Vec<f64>) -> f64 {
        match self.constraint_rules.constrained {
            OptimizationType::Unconstrained => {
                self.cost_objective_function(x)
            },
            OptimizationType::ConstrainedProgramming => {
                self.cost_objective_function(x)
            }
            OptimizationType::ConstrainedMerit => {
                let objective_function_cost = self.cost_objective_function(x);
                let constraints_cost = self.cost_value_constraints(x);
                let total_cost = objective_function_cost + constraints_cost;
                total_cost
            },
        }
    }

    pub fn cost_at_x_exclusive_objective_function(&mut self, x: &Vec<f64>) -> f64 {
        self.cost_objective_function(x)
    }

    pub fn constraint_values_at_x_no_rules_individual(&mut self, x: &Vec<f64>) -> Row<f64> {
        let constraint_values: Vec<f64> = 
            (0..self.n_constraints())
                .map(|index| {
                    self.inequality_contraint_value(index, x)
                })
                .collect();
        Row {
            cells: constraint_values
        }
    }

    pub fn cost_at_pos(&mut self, pos: &PosNDof<f64>) -> f64 {
        self.cost_at_x(&pos.x)
    }

    pub fn itterate_constraint_rules(&mut self) -> & mut Self {
        self.constraint_rules.magnification_factor =
            self.constraint_rules.magnification_factor * self.constraint_rules.magnification_factor_increment_factor 
            + self.constraint_rules.magnification_factor_increment_addition;

        self
    }

    pub fn derivative_objective_function(&mut self, coordinates: &Vec<f64>) -> SpatialVectorNDof<f64, IsColl> {

        let differentation_method = self.optimization_settings.differentiation_rule;

        let n_dof = coordinates.len();
        let mut objective_gradients = SpatialVectorNDof::new_from_direction(vec![0.; n_dof]);

        let search_vector = SpatialVectorWithBasePointNDof::new_with_base(&coordinates, &vec![0.; n_dof]);
        let step_size = self.optimization_settings.differentiating_step_size;


        match differentation_method {
            DifferentiationRule::CentralDifference => {
                for i_dof in 0.. n_dof {
                    let cost_objective_function_at_small_step_forward = self.cost_at_x_exclusive_objective_function(&search_vector.volatile().set_vector_i_value(i_dof, step_size).end_position_vec());
                    let cost_objective_function_at_small_step_bacward = self.cost_at_x_exclusive_objective_function(&search_vector.volatile().set_vector_i_value(i_dof, -step_size).end_position_vec());
                    let cost_difference_objective_function = cost_objective_function_at_small_step_forward - cost_objective_function_at_small_step_bacward;
                    let gradient_objective_function = cost_difference_objective_function / (2. * step_size);
                    objective_gradients.vector[i_dof] = gradient_objective_function;
                }
            },
            DifferentiationRule::OneSided => {

                let cost_current_position = self.cost_at_x(&coordinates);

                for i_dof in 0.. n_dof {

                    let cost_objective_function_at_small_step_forward = self.cost_at_x_exclusive_objective_function(&search_vector.volatile().set_vector_i_value(i_dof, step_size).end_position_vec());
                    let cost_difference_objective_function = cost_objective_function_at_small_step_forward - cost_current_position;
                    let gradient_objective_function = cost_difference_objective_function / step_size;
                    objective_gradients.vector[i_dof] = gradient_objective_function;

                }
            },
        }

        objective_gradients
    }

    pub fn derivative_all_inequality_constraints(&mut self, coordinates: &Vec<f64>) -> Vec<SpatialVectorNDof<f64, IsColl>> {
        let mut derivatives_all_constraints = Vec::<SpatialVectorNDof<f64, IsColl>>::new();
        for inequality_constraint_number in  0.. self.n_constraints() {
            derivatives_all_constraints.push(
                self.derivative_inequality_constraint(inequality_constraint_number, coordinates)
            )
        }
        derivatives_all_constraints
    }

    pub fn derivative_inequality_constraint(&mut self, inequality_constraint_number: usize, coordinates: &Vec<f64>) -> SpatialVectorNDof<f64, IsColl> {
        let differentation_method = self.optimization_settings.differentiation_rule;

        let n_dof = coordinates.len();
        let mut constraint_gradients = SpatialVectorNDof::new_from_direction(vec![0.; n_dof]);

        let search_vector = SpatialVectorWithBasePointNDof::new_with_base(&coordinates, &vec![0.; n_dof]);
        let step_size = self.optimization_settings.differentiating_step_size;


        match differentation_method {
            DifferentiationRule::CentralDifference => {
                for i_dof in 0.. n_dof {
                        let cost_contraint_functions_at_small_step_forward = 
                            self.inequality_contraint_value(inequality_constraint_number, &search_vector.volatile().set_vector_i_value(i_dof, step_size).end_position_vec());
                        let cost_contraint_functions_at_small_step_bacward = 
                            self.inequality_contraint_value(inequality_constraint_number, &search_vector.volatile().set_vector_i_value(i_dof, -step_size).end_position_vec());
                        let cost_difference_contraint_functions = cost_contraint_functions_at_small_step_forward - cost_contraint_functions_at_small_step_bacward;
                        constraint_gradients.vector[i_dof] = cost_difference_contraint_functions / (2. * step_size);
                }
            },
            DifferentiationRule::OneSided => {

                let cost_current_position = self.cost_at_x(&coordinates);

                for i_dof in 0.. n_dof {

                    let cost_objective_function_at_small_step_forward = 
                        self.inequality_contraint_value(inequality_constraint_number, &search_vector.volatile().set_vector_i_value(i_dof, step_size).end_position_vec());
                    let cost_difference_objective_function = cost_objective_function_at_small_step_forward - cost_current_position;
                    constraint_gradients.vector[i_dof] = cost_difference_objective_function / step_size;

                }
            },
        }

        constraint_gradients

    }
}

impl CostController {
    fn cost_value_constraints(&mut self, x: &Vec<f64>) -> f64 {
        let mut total_cost = 0.;

        for constraint_index in 0..self.n_constraints() {
            total_cost += self.cost_value_constraint_at_index(constraint_index, x);
        }

        total_cost
    }

    pub fn cost_value_constraint_at_index(&mut self, constraint_index: usize, x: &Vec<f64>) -> f64 {
        let constraint_value = self.inequality_contraint_value(constraint_index, x);
            
        match self.constraint_rules.penalty_function.unwrap() {
            PenaltyFunction::Exterior(exterior_penalty_function) => {
                match exterior_penalty_function {
                    ExteriorPenaltyFunction::Quadratic => {
                        self.quadratic_penalty_value(constraint_value)
                    },
                }
            },
            PenaltyFunction::Interior(interior_penalty_function) => {
                match  interior_penalty_function {
                    InteriorPenaltyFunction::LogBarrier => {
                        self.log_barrier_value(constraint_value)
                    },
                    InteriorPenaltyFunction::InverseLogBarrier => {
                        self.inverse_barrier_value(constraint_value)
                    },
                }
            }
        }
    }

    pub fn penalty_functions_derivatives_separated(&mut self, x: &Vec<f64>) -> Collumn<f64> {
        let mut collumn = Collumn::new_with_constant_values(self.n_constraints(), 0.);

        for constraint_index in 0..self.n_constraints() {
            let constraint_value = self.inequality_contraint_value(constraint_index, x);
            
            match self.constraint_rules.penalty_function.unwrap() {
                PenaltyFunction::Exterior(exterior_penalty_function) => {
                    match exterior_penalty_function {
                        ExteriorPenaltyFunction::Quadratic => {
                            collumn[constraint_index] = self.quadratic_penalty_derivative_value(constraint_value)
                        },
                    }
                },
                PenaltyFunction::Interior(interior_penalty_function) => {
                    match interior_penalty_function {
                        InteriorPenaltyFunction::LogBarrier => {
                            collumn[constraint_index] = self.log_barrier_derivative_value(constraint_value);
                        },
                        InteriorPenaltyFunction::InverseLogBarrier => {
                            collumn[constraint_index] = self.inverse_barrier_derivative_value(constraint_value);
                        },
                    }
                },
            }
        }

        collumn
    }

    pub fn constraints_are_met(&mut self, coordinates: &Vec<f64>, tolerance_value: f64) -> bool {
        let mut constraints_met = true;      
        for constraint_number in 0..self.n_constraints() {
            let constraints_value = self.inequality_contraint_value(constraint_number, coordinates) - tolerance_value;
            if constraints_value > 0. {
                constraints_met = false
            };
        }
        constraints_met
    }
}

impl CostController {
    fn cost_objective_function(&mut self, x: &Vec<f64>) -> f64 {
        self.number_cost_function_evaluation += 1;
        let result = (self.cost_function)(x);
        result.function_cost
    }

    pub fn inequality_contraint_value(&mut self, inequality_constraint_index: usize, x: &Vec<f64>) -> f64 {
        if inequality_constraint_index < self.constraint_inequality_functions.len() {
            self.number_constraint_function_evaluations += 1;
            (self.constraint_inequality_functions[inequality_constraint_index].constraint_function)(x)
        } else {
            self.number_cost_function_evaluation += 1;
            (self.cost_function)(x).model_constraints[inequality_constraint_index-self.constraint_inequality_functions.len()].value
        }

    }

    fn quadratic_penalty_value(&self, constraint_value: f64) -> f64 {
        if constraint_value < 0. {
            0.
        } else {
            self.constraint_rules.magnification_factor /2. * constraint_value.powi(2)
        }
    }

    fn quadratic_penalty_derivative_value(&self, constraint_value: f64) -> f64 {
        if constraint_value < 0. {
            0.
        } else {
            self.constraint_rules.magnification_factor * constraint_value
        }
    }


    fn log_barrier_value(&self, constraint_value: f64) -> f64 {
        if constraint_value > 0. {
            1.*self.constraint_rules.magnification_factor
        } else {
            - 1. / self.constraint_rules.magnification_factor * (-constraint_value).ln()
        }
    }

    fn log_barrier_derivative_value(&self, constraint_value: f64) -> f64 {
        if constraint_value > 0. {
            0.
        } else {
            - 1. / self.constraint_rules.magnification_factor * 1. / constraint_value
        }
    }

    fn inverse_barrier_value(&self, constraint_value: f64) -> f64 {
        if constraint_value > 0. {
            1.*self.constraint_rules.magnification_factor
        } else {
            1. / self.constraint_rules.magnification_factor * (- 1. / constraint_value)
        }
    }

    fn inverse_barrier_derivative_value(&self, constraint_value: f64) -> f64 {
        if constraint_value > 0. {
            0.
        } else {
             1. / self.constraint_rules.magnification_factor * 1. / constraint_value.powi(2)
        }
    }
}