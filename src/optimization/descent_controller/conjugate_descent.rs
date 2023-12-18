use crate::{*};

impl DescentController {
    pub fn use_conjugate_descent (&mut self, conjugate_method: ConjugateGradient) -> &mut Self  {

        let current_gradient = self.current_gradient.clone();
        let previous_gradient = self.previous_gradient.clone().unwrap();

        let previous_search_line_direction = self.previous_search_line.clone().unwrap().vector();

        let conjugate_gradient_factor: f64;
        let spiraling_condition = 
            previous_gradient.transpose()* &current_gradient 
            > 0.2 * current_gradient.transpose() * &current_gradient;

        // This matches the spiraling condition to it's consequences
        match spiraling_condition {
            false => {
                // The spiraling condition is false so a normal...
                // ... calculation can be done.
                match conjugate_method {
                    ConjugateGradient::FletcherReeves => {

                        conjugate_gradient_factor = 
                            current_gradient.transpose() * &current_gradient 
                                / (previous_gradient.transpose() * &previous_gradient);
                    },

                    ConjugateGradient::PolakRibiere => {
                        conjugate_gradient_factor = 
                            (&current_gradient - &previous_gradient).transpose()*&current_gradient 
                                / (previous_gradient.transpose()*&previous_gradient)
                    },
                }
            },
            true => {
                // println!("Spiraling detected!");
                conjugate_gradient_factor = 0.;
            }
        }

        let previous_decent_step_scaled = conjugate_gradient_factor * self.previous_search_line.clone().unwrap().vector();
        let conjugate_decent_gradient = -1. * &current_gradient + previous_decent_step_scaled;

        self.current_search_line = Option::Some(
            SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &conjugate_decent_gradient.vector)
        );

        self
    }
}