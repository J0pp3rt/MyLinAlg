use crate::{*};

#[derive(Clone)]
pub enum CostRule {
    Unconstrained,
    Constrained(ConstrainedCostRule)
}

#[derive(Clone)]
pub enum ConstrainedCostRule {
    ExteriorQuadratic,
    InteriorLog,
    InteriorInverseBarrier,
}

pub struct CostController {
    cost_function: CostFunction,
    cost_function_normalization_factor: Vec<f64>,
    constraint_equality_functions: Vec<CostFunction>,
    constraint_inequality_functions: Vec<InequalityConstraint>,
    number_cost_function_evaluation: usize,
    number_constraint_function_evaluations: usize,
    cost_rule: CostRule,
}

impl CostController {
    pub fn new(
        cost_function: CostFunction, 
        cost_function_normalization_factor: Vec<f64>,
        cost_function_algebraic_derivative_supplied: bool,
        cost_function_algebraic_derivative: Option<AlgebraicDerivative>,
        constraint_equality_functions: Vec<CostFunction>,
        constraint_inequality_functions: Vec<InequalityConstraint>,
        cost_rule: CostRule,
    ) -> Self {

        Self { 
            cost_function, 
            cost_function_normalization_factor, 
            constraint_equality_functions, 
            constraint_inequality_functions,
            number_cost_function_evaluation: 0,
            number_constraint_function_evaluations: 0,
            cost_rule,
        }
    }
}

impl CostController {
    pub fn cost_at_x(&mut self, x: &Vec<f64>) -> f64 {
        match self.cost_rule {
            CostRule::Unconstrained => {
                self.number_cost_function_evaluation += 1;
                (self.cost_function)(x)
            },
            CostRule::Constrained(_) => todo!(),
        }
    }

    pub fn cost_at_pos(&mut self, pos: &PosNDof<f64>) -> f64 {
        self.cost_at_x(&pos.x)
    }
}