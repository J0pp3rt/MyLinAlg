use crate::{*};

impl DescentController {
    
    pub fn use_quadratic_programming_method(&mut self, quasi_newton_method: QuasiNewton) -> &mut Self {
        assert!(if let Option::Some(_) = self.previous_hessian_approx {true} else {false}, "No Hessian approximation available from last itteration");
        
        let comparison_epsilon = 0.00000001;
        self.use_quasi_newton_method(quasi_newton_method, false);

        let mut active_constraint_numbers = Vec::new();
        let mut active_constraint_values = Vec::new();
        let mut active_constraint_derivatives = Vec::new();
        for constraint_number in 0 .. self.line_searcher.cost_controller.n_constraints() {
            let constraint_cost = self.line_searcher.cost_controller.inequality_contraint_value(constraint_number, &self.current_coordinates.x());
            println!("Constraint value: {}", constraint_cost);
            if constraint_cost > 0. {
                active_constraint_numbers.push(constraint_number);
                active_constraint_values.push(constraint_cost);
                active_constraint_derivatives.push(self.line_searcher.cost_controller.derivative_inequality_constraint(constraint_number, &self.current_coordinates.x()))
            }
        }
        let n_active_constraints = active_constraint_numbers.len();

        let current_hessian = self.current_hessian_approx.clone().unwrap();
        let mut local_gradient_objective_spatial_vec = self.line_searcher.cost_controller.derivative_objective_function( &self.current_coordinates.x());
        let normalization_factor = local_gradient_objective_spatial_vec.length() / self.first_gradient_length;
        println!("normalization factor {:}", normalization_factor);
        let local_gradient_objective = local_gradient_objective_spatial_vec.normalize().scale(normalization_factor).clone().to_collumn();
        let solved_system: Solved2D<f64>;
        let mut search_line_vec = Vec::<f64>::new();

        if n_active_constraints == 0 {
            println!("No constraints");
            let search_line = -1. *current_hessian*local_gradient_objective;
            search_line_vec = search_line.cells;
        } else {
            println!("constraints used!");
            let mut C = Matrix::new_with_constant_values(self.n_dof(), n_active_constraints, 0.);
            for (active_constraint_number, constraint_derivative) in active_constraint_derivatives.iter().enumerate() {
                for i_dof in 0..self.n_dof() {
                    C[i_dof][active_constraint_number] = constraint_derivative.vector[i_dof];
                }
            }
            let mut C_t = C.clone();
            C_t.transpose_non_skinny();

            let helper = C.clone() * (C_t.clone() * current_hessian.clone() * C.clone()).inverse_through_gauss();

            
            let eye = Matrix::new_square_eye(self.n_dof(), 1.);
            let active_constraint_values_vec = SpatialVectorNDof::new_from_direction(active_constraint_values);

            // let search_line_vec = 
            //     -1. * current_hessian_inverse.clone()
            //     * (eye - C * (C_t * current_hessian_inverse.clone()*C).inverse_through_gauss() * C_t * current_hessian_inverse) * local_gradient_objective
            //     - current_hessian_inverse * C * (C_t * current_hessian_inverse * C).inverse_through_gauss() * active_constraint_values; 

            let search_line = 
                -1. * current_hessian.clone() * (eye.clone() - helper.clone() * C_t.clone() * current_hessian.clone()) * local_gradient_objective.clone()
                + (-1.) * current_hessian.clone() * helper.clone() * active_constraint_values_vec;
            let search_line_no_constraint = -1. *current_hessian.clone()*local_gradient_objective.clone();
            let check_1 = eye.clone() - helper.clone() * C_t.clone() * current_hessian.clone();
            let check_2 = current_hessian.clone() * (eye.clone() - helper.clone() * C_t.clone() * current_hessian.clone()) * local_gradient_objective.clone();
            search_line_vec = search_line.vector;
            // let step_1 = active_constraint_derivative_matrix_transposed.clone() * active_constraint_derivative_matrix.clone();
            // println!("step_1 done");
            // let step_2 = step_1.inverse_through_gauss();
            // println!("step_2 done");
            // let step_3 = -1. * step_2;
            // println!("step_3 done");
            // let step_4  = step_3 * active_constraint_derivative_matrix.clone().transpose_non_skinny();
            // println!("step_4 done");
            // let step_5 = step_4 * local_gradient_objective.clone();
            // println!("step_5 done");
    
            // let mu_value: Collumn<f64>;
            // if n_active_constraints == 0 {
            //     mu_value = Collumn { cells: vec![0. ; self.n_dof()]};
            // } else {
            //     mu_value = 
            //     -1. * (active_constraint_derivative_matrix_transposed.clone() * active_constraint_derivative_matrix.clone()).inverse_through_gauss()
            //     * active_constraint_derivative_matrix.clone().transpose_non_skinny() * local_gradient_objective.clone();
            // }
            // let mut a_matrix = Matrix::new_square_with_constant_values(self.n_dof()+n_active_constraints, 0.);
            
            // a_matrix.update(0..self.n_dof(), 0..self.n_dof(), current_hessian);
            // a_matrix.update(0..self.n_dof(), self.n_dof()..self.n_dof()+n_active_constraints, active_constraint_derivative_matrix.clone());
            // a_matrix.update(self.n_dof()..self.n_dof()+n_active_constraints, 0.. self.n_dof(), active_constraint_derivative_matrix_transposed.clone());
    
            // let mut b_matrix = Matrix::new_with_constant_values(self.n_dof()+n_active_constraints, 1, 0.);
    
            // let active_constraint_values_collumn = Collumn {
            //     cells: active_constraint_values
            // };
    
            // b_matrix.update(0.. self.n_dof(), 0, local_gradient_objective + active_constraint_derivative_matrix*mu_value);
            // b_matrix.update(self.n_dof().. self.n_dof()+n_active_constraints, 0, active_constraint_values_collumn);
            // b_matrix = b_matrix *-1.;
    
            // let solver_2d = Solver2D{
            //     A_matrix: a_matrix,
            //     B_matrix: b_matrix,
            //     solver: Solver2DStrategy::Guass,
            // };

            // let solved_system = Matrix::new_from_solver(solver_2d);
            
            // for index in 0..self.n_dof() {
            //     search_line_vec.push(
            //         solved_system.B_matrix[index][0]
            //     );
            // }
        }
        

        let search_line = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &search_line_vec);

        self.current_search_line = Option::Some(search_line);

        self
    }

}