use crate::{*};

pub fn get_cost_function_springs(x: &Vec<f64>) -> f64 {
    let l0: f64 = 10.;
    let k1: f64 = 8.;
    let k2: f64 = 4.;
    let p1: f64 = 50.;
    let p2: f64 = 100.;

    let position_node_1 = PosNDof::new(vec![0., l0]);
    let position_node_2 = PosNDof::new(vec![0., -l0]);

    let dl = |position_node: PosNDof<f64>, x: Vec<f64>| {
        ((x[0] - position_node.x[0]).powi(2) + (x[1] - position_node.x[1]).powi(2)).sqrt() - l0
    };

    let cost = 0.5 * k1 * dl(position_node_1.clone(), x.clone()).powi(2)
        + 0.5 * k2 * dl(position_node_2.clone(), x.clone()).powi(2)
        - p1 * x[0]
        - p2 * x[1];

    cost
}


pub fn get_cost_function_roosenbruck(x: &Vec<f64>) -> f64 {
    let a = 1.;
    let b = 100.;

    (a - x[0]).powi(2) + b*(x[1] - x[0].powi(2)).powi(2)
}