use crate::{*};

impl DescentController {
    pub fn use_steepest_descent (&mut self) -> &mut Self {
        let local_gradient = self.current_gradient.clone();
        let search_direction = -1. * local_gradient;
        self.current_search_line = Option::Some(
            SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &search_direction.vector)
        );
        self
    }
}