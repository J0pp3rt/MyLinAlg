use crate::{*};

impl DescentController {
    
    pub fn use_quasi_newton_method(&mut self, quasi_newton_method: QuasiNewton) -> &mut Self {
        assert!(if let Option::Some(_) = self.previous_hessian_approx {true} else {false}, "No Hessian approximation available from last itteration");

        let n_dof = self.n_dof();

        let coordinates_previous_step = self.previous_coordinates.clone().unwrap().as_vector();
        let coordinates_current_step = self.current_coordinates.as_vector();
        let dx = coordinates_current_step - coordinates_previous_step;

        let gradient_previous_step = self.previous_gradient.clone().unwrap();
        let gradient_current_step = self.current_gradient.clone();
        let y = gradient_current_step.clone() - gradient_previous_step;

        let hessian_approx = self.previous_hessian_approx.clone().unwrap();
        let sigma = 1. / (y.transpose()*&dx);

        let search_line: SpatialVectorWithBasePointNDof<f64, crate::IsColl>;
        match quasi_newton_method {
            QuasiNewton::BFGS => {
                let i = Matrix::new_square_eye(n_dof, 1.);
        
                let next_hessian = (&i - sigma*&dx*y.transpose()) * hessian_approx * (&i - sigma*&y*dx.transpose()) + sigma*&dx*dx.transpose();
                
                self.current_hessian_approx = Option::Some(next_hessian.clone());
                let search_direction = -1. * next_hessian * gradient_current_step;
                search_line = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &search_direction.vector);
            },
            QuasiNewton::BFGSBeforeInverse => {                
                let da = 
                    sigma * &y*y.transpose() 
                    -  &hessian_approx*&dx*dx.transpose()*&hessian_approx / (dx.transpose() * &hessian_approx * &dx);
        
                let next_hessian = hessian_approx + da;
        
                self.current_hessian_approx = Option::Some(next_hessian.clone());
                let next_hessian_inverse = next_hessian.inverse_through_gauss();
                let search_direction = -1. * next_hessian_inverse * gradient_current_step;
                search_line = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &search_direction.vector);
            },
            QuasiNewton::BFGSBeforeInverseMoreStable => {
                let da = 
                    (1. + sigma * dx.transpose()*&hessian_approx*&dx)*(sigma * &y*y.transpose())
                    - sigma * &y*dx.transpose()*&hessian_approx 
                    - sigma * &hessian_approx*&dx*y.transpose();
        
                let next_hessian = hessian_approx + da;
        
                self.current_hessian_approx = Option::Some(next_hessian.clone());
                let next_hessian_inverse = next_hessian.inverse_through_gauss();
                let search_direction = -1. * next_hessian_inverse * gradient_current_step;
                search_line = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &search_direction.vector);
            },
            QuasiNewton::DFP => {
                let db = 
                    sigma * &dx*dx.transpose()
                    - &hessian_approx*&y*y.transpose()*&hessian_approx / (y.transpose()*&hessian_approx*&y);
        
                let next_hessian = hessian_approx + db;
        
                self.current_hessian_approx = Option::Some(next_hessian.clone());
                let search_direction = -1. * next_hessian * gradient_current_step;
                search_line = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &search_direction.vector);
            },
            QuasiNewton::DFPMoreStable => {
                let db = 
                    (1. + sigma * y.transpose()*&hessian_approx*&y) * (sigma * &dx*dx.transpose())
                    - sigma * &dx*y.transpose()*&hessian_approx
                    - sigma * &hessian_approx*&y*dx.transpose();

                let next_hessian = hessian_approx + db;

                self.current_hessian_approx = Option::Some(next_hessian.clone());
                let search_direction = -1. * next_hessian * gradient_current_step;
                search_line = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &search_direction.vector);
            },
        }

        self.current_search_line = Option::Some(search_line);
        
        self

    }
}
