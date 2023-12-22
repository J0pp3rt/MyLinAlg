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
    ) -> Self {

        Self { 
            cost_function, 
            cost_function_normalization_factor, 
            constraint_equality_functions, 
            constraint_inequality_functions,
            number_cost_function_evaluation: 0,
            number_constraint_function_evaluations: 0,
            constraint_rules,
        }
    }
}

impl CostController {
    pub fn cost_at_x(&mut self, x: &Vec<f64>) -> f64 {
        match self.constraint_rules.constrained {
            OptimizationType::Unconstrained => {
                self.cost_objective_function(x)
            },
            OptimizationType::Constrained => {
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
            (0..self.constraint_inequality_functions.len())
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
}

impl CostController {
    fn cost_value_constraints(&mut self, x: &Vec<f64>) -> f64 {
        let mut total_cost = 0.;

        for constraint_index in 0..self.constraint_inequality_functions.len() {
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
        let mut collumn = Collumn::new_with_constant_values(self.constraint_inequality_functions.len(), 0.);

        for constraint_index in 0..self.constraint_inequality_functions.len() {
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
}

impl CostController {
    fn cost_objective_function(&mut self, x: &Vec<f64>) -> f64 {
        self.number_cost_function_evaluation += 1;
        (self.cost_function)(x)
    }

    pub fn inequality_contraint_value(&mut self, inequality_constraint_index: usize, x: &Vec<f64>) -> f64 {
        self.number_constraint_function_evaluations += 1;
        (self.constraint_inequality_functions[inequality_constraint_index].constraint_function)(x)
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