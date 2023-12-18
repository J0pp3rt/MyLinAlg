use crate::{*};

pub type CostFunction = Box<dyn Fn(&Vec<f64>) -> f64 + 'static >;
pub type AlgebraicDerivative = Box<dyn Fn(Vec<f64>) -> Vec<f64> + 'static >;

#[derive(Clone)]
pub enum DifferentiationRule {
    CentralDifference,
    OneSided
}
pub struct InequalityConstraint {
    pub constraint_function: CostFunction,
    pub algebraic_derivative_supplied: bool,
    pub algebraic_derivative: Option<AlgebraicDerivative>,
}

pub struct Optimizer {
    descent_controller: DescentController,
    coordinates_history: Vec<Vec<f64>>,
    convergence_history: Vec<f64>,
}

impl Optimizer {
    pub fn new(
        descent_controller: DescentController,
        coordinates_history: Vec<Vec<f64>>, 
        convergence_history: Vec<f64>
    ) -> Self {
        Optimizer {
            descent_controller,
            coordinates_history,
            convergence_history,
        }
    }
}

impl Optimizer {
    pub fn run_optimization(&mut self) {
        
    }
}