use crate::{*};

impl DescentController {

    pub fn use_non_linear_programming_method(&mut self, quasi_newton_method: QuasiNewton) -> &mut Self {

        // self.update_objective_hessian(quasi_newton_method);
        self.use_quasi_newton_method(quasi_newton_method, false);
        self.update_constraint_hessian(quasi_newton_method);

        let n_constraint = self.line_searcher.cost_controller.n_constraints();

        let constraint_gradients = self.current_constraint_gradient.clone().unwrap();
        // let objective_hessian = self.non_linear_logger.active_hessian.clone();
        let objective_hessian = self.current_hessian_approx.clone().unwrap();
        let current_gradient = self.current_gradient.clone();
        let constraint_hessians = self.current_constraint_hessian_approx.clone().unwrap();

        let mut active_constraint_gradients = Vec::<SpatialVectorNDof<f64, IsColl>>::new();
        let mut active_constraint_mus = Vec::<f64>::new();
        let mut active_constraint_values = Vec::<f64>::new();
        let mut n_active_constraints: usize = 0;

        let constraint_values = self.line_searcher.cost_controller.constraint_values_at_x_no_rules_individual(&self.current_coordinates.x);

        let mut hessian_part = objective_hessian.clone();

        for (constraint_index, ((constraint_gradient, constraint_hessian), constraint_value)) in constraint_gradients.iter().zip(constraint_hessians).zip(constraint_values.cells).enumerate() {
            // let build_off_range = 0.5;
            // if constraint_value > 0.  || constraint_value > -build_off_range {
                if constraint_value > -0.01 {
                // println!("~~--~~--~~--~~--~~");
                // println!("constraint used!, value = {}, gradient = {:?}", constraint_value, constraint_gradient.vector);
                // println!("~~--~~--~~--~~--~~");
                n_active_constraints += 1;
                active_constraint_values.push(constraint_value);
                active_constraint_gradients.push(constraint_gradient.clone());
                
                let c_constraint = constraint_gradient.clone().to_collumn();

                // let mu_correction_factor: f64;
                // if constraint_value < 0. {
                //     mu_correction_factor = (build_off_range + constraint_value) / build_off_range
                // } else {
                //     mu_correction_factor = 1.;
                // }

                let mu = -1. / (c_constraint.clone().transpose() * c_constraint.clone()) * c_constraint.clone().transpose() * current_gradient.clone() ;
                active_constraint_mus.push(mu);

                hessian_part = hessian_part + mu * constraint_hessian / n_active_constraints as f64;

            }
        }

        // if n_active_constraints == 0  {
        //     println!("oooooooooooooooooooooooo");
        //     println!("normal unconstraint!");
        //     println!("oooooooooooooooooooooooo");

        //     let search_direction = -1. * objective_hessian * self.current_gradient.clone();
        //     self.current_search_line = Option::Some(SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &search_direction.vector));
        // } else {

            let mut active_constraint_gradient_matrix = Matrix::new_with_constant_values(self.n_dof(), n_active_constraints, 0.);
            
            for (active_constraint_number, active_constraint_gradient) in active_constraint_gradients.iter().enumerate() {
                for i_dof in 0..self.n_dof() {
                    active_constraint_gradient_matrix[i_dof][active_constraint_number] = active_constraint_gradient.vector[i_dof]
                }
            }

            let active_constraint_mus_collumn = Collumn::new_form_vec(active_constraint_mus.clone());

            let mut lhs_matrix = Matrix::new_with_constant_values(self.n_dof()+n_active_constraints, self.n_dof()+n_active_constraints, 0.);
            lhs_matrix.update(0..self.n_dof(), 0..self.n_dof(), hessian_part);

            let mut rhs_values: Vec<f64>;
            if n_active_constraints != 0 {
                lhs_matrix.update(0..self.n_dof(), self.n_dof()..self.n_dof()+n_active_constraints, active_constraint_gradient_matrix.clone());
                let active_matrix_transposed = active_constraint_gradient_matrix.clone().transpose_non_skinny().clone();
                lhs_matrix.update(self.n_dof()..self.n_dof()+n_active_constraints, 0.. self.n_dof(), active_matrix_transposed);
                rhs_values = (self.current_gradient.clone() + active_constraint_gradient_matrix * active_constraint_mus_collumn).vector;
                rhs_values.append(&mut active_constraint_values);
            } else {
                rhs_values = (self.current_gradient.clone()).vector;
            }

            let rhs_matrix = -1. * Matrix::new_from_vector_is_collumn(rhs_values);

            let system_to_solve = Solver2D {
                A_matrix: lhs_matrix,
                B_matrix: rhs_matrix,
                solver: Solver2DStrategy::Guass
            };

            let solved_system = Matrix::new_from_solver(system_to_solve);

            let search_direction_uncorrected = solved_system.B_matrix.to_vec_row_major();
            let search_direction_uncorrected_spatial = SpatialVectorNDof::new_from_direction(search_direction_uncorrected[0..self.n_dof()].to_vec());

            let search_direction_corrected: Vec<f64>;
            let allowed_max_distance_per_step: f64;

            // if n_active_constraints == 0 {
            //     allowed_max_distance_per_step = 1.;
            // } else {
            //     allowed_max_distance_per_step = 10.;
            // }

            // if search_direction_uncorrected_spatial.length() > allowed_max_distance_per_step {
            //     search_direction_corrected = search_direction_uncorrected_spatial.clone().normalize().scale(allowed_max_distance_per_step).clone().vector;
            // } else {
            //     search_direction_corrected = search_direction_uncorrected_spatial.vector;
            // }

            // let search_direction = search_direction_corrected;

            let stepsize_normalization_factor = (1. + n_active_constraints as f64) ;

            let search_direction = (search_direction_uncorrected_spatial / (stepsize_normalization_factor)).vector;
            // let search_direction = search_direction_uncorrected_spatial.vector;

            let search_line = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &search_direction);

            self.current_search_line = Option::Some(search_line);

            // let mut all_constraint_contributions = Collumn::new_form_vec(vec![0.; self.n_dof()]);

            // for (constraint_number, constraint_gradient) in constraint_gradients.iter().enumerate() {
            //     let c_constraint = constraint_gradient.clone().to_collumn();
            //     let mu = -1. / (c_constraint.clone().transpose() * c_constraint.clone()) * c_constraint.clone().transpose() * self.local_gradient();
            //     for i_dof in 0..self.n_dof() {
                    
            //         all_constraint_contributions[i_dof] += constraint_gradient.vector[i_dof] * mu;
                
                    
            //     }
            // }

            // let mut all_constraint_contributions = SpatialVectorNDof::new_from_direction(vec![0.; self.n_dof()]);
            // for (gradients, mu) in active_constraint_gradients.iter().zip(active_constraint_mus) {
            //     all_constraint_contributions = all_constraint_contributions + gradients * mu;
            // }

            // let end_condition = current_gradient + all_constraint_contributions;
            // let end_condition = SpatialVectorNDof::new_from_direction(all_constraint_contributions.cells);

            // self.non_linear_logger.end_condition = end_condition.length();

            
            // if self.descent_logger.itteration_counter % 10 == 0 {
            //     // self.current_hessian_approx = Option::Some(Matrix::new_square_eye(self.n_dof(), 1.));
            //     // for constraint_hessian_vec in self.clone().current_constraint_hessian_approx.iter_mut() {
            //     //     for constraint_hessian in constraint_hessian_vec.iter_mut() {
            //     //         *constraint_hessian = Matrix::new_square_eye(self.n_dof(), 1.);
            //     //     }
            //     // }
            //     println!("resetting all hessians!")
            // }
            // if n_active_constraints == 0 && self.descent_logger.last_iteration_was_constraint {
            //     self.current_hessian_approx = Option::Some(Matrix::new_square_eye(self.n_dof(), 1.));
            // }

        // }
        self
    }

    pub fn non_linear_derivative(&mut self) -> f64 {

        let constraint_values = self.line_searcher.cost_controller.constraint_values_at_x_no_rules_individual(&self.current_coordinates.x);
        
        let constraint_gradients = self.current_constraint_gradient.clone().unwrap();
        let current_gradient = self.current_gradient.clone();

        let mut all_constraint_contributions = Collumn::new_form_vec(vec![0.; self.n_dof()]);

        let mut n_active_constraints: usize = 0;

        for (constraint_number, (constraint_gradient, constraint_value)) in constraint_gradients.iter().zip(constraint_values.cells).enumerate() {
            if constraint_value > -0.1 {
                n_active_constraints += 1;
                let c_constraint = constraint_gradient.clone().to_collumn();
                let mu = -1. / (c_constraint.clone().transpose() * c_constraint.clone()) * c_constraint.clone().transpose() * current_gradient.clone();
                for i_dof in 0..self.n_dof() {
                    
                    all_constraint_contributions[i_dof] += constraint_gradient.vector[i_dof] * mu;
                
                    
                }
            }
        }

        // let mut all_constraint_contributions = SpatialVectorNDof::new_from_direction(vec![0.; self.n_dof()]);
        // for (gradients, mu) in active_constraint_gradients.iter().zip(active_constraint_mus) {
        //     all_constraint_contributions = all_constraint_contributions + gradients * mu;
        // }

        let mut end_condition: SpatialVectorNDof<f64, IsColl>;
        if n_active_constraints == 0 {
            end_condition = current_gradient;
        } else {
            end_condition = current_gradient + all_constraint_contributions;
        }

        let end_condition_normalized = end_condition.clone()  * (end_condition.length() / self.non_linear_logger.end_condition_normalization_factor);

        // let end_condition = current_gradient + all_constraint_contributions;
        // let end_condition = SpatialVectorNDof::new_from_direction(all_constraint_contributions.cells);

        self.non_linear_logger.end_condition = end_condition_normalized.length();

        end_condition_normalized.length()
    }

    fn update_objective_hessian(&mut self, quasi_newton_method: QuasiNewton) -> &mut Self {
        // assert!(if let Option::Some(_) = self.previous_constraint_gradient {true} else {false}, "No contraint gradients available from last itteration");

        let n_dof = self.n_dof();

        if self.non_linear_logger.cycles_left_on_active_before_switch == 0 {
            self.non_linear_logger.active_hessian = self.non_linear_logger.secondary_hessian.clone();
            self.non_linear_logger.secondary_hessian = Matrix::new_square_eye(self.n_dof(), 1.);
            self.non_linear_logger.cycles_left_on_active_before_switch = self.non_linear_logger.cycles_per_active_hessian;
            self.non_linear_logger.cycles_left_before_building_secondary = self.non_linear_logger.cycles_per_active_hessian - self.non_linear_logger.cycles_needed_for_building;
        }

        self.non_linear_logger.secondary_building =  self.non_linear_logger.cycles_left_before_building_secondary <= 0;

        let coordinates_previous_step = self.previous_coordinates.clone().unwrap().as_vector();
        let coordinates_current_step = self.current_coordinates.as_vector();
        let dx = coordinates_current_step - coordinates_previous_step;

        let gradient_previous_step = self.previous_gradient.clone().unwrap();
        let gradient_current_step = self.current_gradient.clone();
        let y = gradient_current_step.clone() - gradient_previous_step;

        let hessian_approx_active = self.non_linear_logger.active_hessian.clone();
        let build_secondary = self.non_linear_logger.secondary_building;

        let sigma = 1. / (y.transpose()*&dx);

        let search_line: SpatialVectorWithBasePointNDof<f64, crate::IsColl>;

        match quasi_newton_method {
            QuasiNewton::BFGS => {
                let i = Matrix::new_square_eye(n_dof, 1.);
        
                let next_hessian = (&i - sigma*&dx*y.transpose()) * hessian_approx_active * (&i - sigma*&y*dx.transpose()) + sigma*&dx*dx.transpose();

                if build_secondary {
                    self.non_linear_logger.secondary_hessian = (&i - sigma*&dx*y.transpose()) * self.non_linear_logger.secondary_hessian.clone() * (&i - sigma*&y*dx.transpose()) + sigma*&dx*dx.transpose();
                }
                
                self.non_linear_logger.active_hessian = next_hessian.clone();
                let search_direction = -1. * next_hessian * gradient_current_step;
                search_line = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &search_direction.vector);
            },
            _ => {},
        }

        self.non_linear_logger.cycles_left_on_active_before_switch += -1;
        self.non_linear_logger.cycles_left_before_building_secondary += -1;

        self
    }


    fn update_constraint_hessian(&mut self, quasi_newton_method: QuasiNewton) -> &mut Self {
        assert!(if let Option::Some(_) = self.previous_constraint_gradient {true} else {false}, "No contraint gradients available from last itteration");

        let n_dof = self.n_dof();
        let n_contraint = self.line_searcher.cost_controller.n_constraints();

        let coordinates_previous_step = self.previous_coordinates.clone().unwrap().as_vector();
        let coordinates_current_step = self.current_coordinates.as_vector();
        let dx = coordinates_current_step.clone() - coordinates_previous_step;

        let previous_hessians = self.previous_constraint_hessian_approx.clone().unwrap();
        let previous_gradients = self.previous_constraint_gradient.clone().unwrap();
        let mut current_gradients = Vec::<SpatialVectorNDof<f64, IsColl>>::new();
        let mut current_hessains = Vec::<Matrix<f64>>::new();

        for (contraint_index, (previous_hessian, previous_gradient)) in previous_hessians.iter().zip(previous_gradients).enumerate() {
            let gradient_current_step = self.line_searcher.cost_controller.derivative_inequality_constraint(contraint_index, &coordinates_current_step.vector);
            current_gradients.push(gradient_current_step.clone());
            let y = gradient_current_step.clone() - previous_gradient;
            let hessian_approx = previous_hessian;
            let sigma = 1. / (y.transpose()*&dx);

            match quasi_newton_method {
                QuasiNewton::BFGSNoInverse => {  
                    let da = 
                    sigma * &y*y.transpose() 
                    -  hessian_approx*&dx*dx.transpose()*hessian_approx / (dx.transpose() * hessian_approx * &dx);
        
                    let next_hessian = hessian_approx + da;
                    
                    current_hessains.push(next_hessian);
                }
                QuasiNewton::BFGSBeforeInverseMoreStable => {
                    let da = 
                        (1. + sigma * dx.transpose()*hessian_approx*&dx)*(sigma * &y*y.transpose())
                        - sigma * &y*dx.transpose()*hessian_approx 
                        - sigma * hessian_approx*&dx*y.transpose();
            
                    let next_hessian = hessian_approx + da;
                    
                    current_hessains.push(next_hessian);
                },
                
                _ => {panic!("Just BFGS method not yet supported in non linear for now!")}
            }

        }

        for current_hessian in current_hessains.iter_mut() {
            let mut contains_nan = false;
            for row in &current_hessian.rows {
                for value in &row.cells {
                    if value.is_nan() {
                        contains_nan = true;
                    }
                }
            }

            if contains_nan {
                *current_hessian = Matrix::new_square_eye(self.n_dof(), 1.);
                println!("Hessian was corrupted!");
            }
        }

        self.current_constraint_hessian_approx = Option::Some(current_hessains);
        self.current_constraint_gradient = Option::Some(current_gradients);



        self
    }

}