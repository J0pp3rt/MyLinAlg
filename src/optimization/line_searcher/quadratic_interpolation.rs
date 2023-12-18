use crate::{*};


impl LineSearcher {
    pub fn quadratic_interpolation_iterate_untill_convergence(&mut self) -> &mut Self {

        while self.quadratic_is_completed().not() {
            self.quadratic_interpolation_iterate();
        }
        self
    }

    fn quadratic_is_completed(&self) -> bool {
        // state guard to guarantee cost values at bounds are known.
        let cost_at_lower_bound: f64;
        let cost_at_upper_bound: f64;
        match (self.cost_lower_bound, self.cost_upper_bound) {
            (Option::Some(cost_lower), Option::Some(cost_upper)) => {
                cost_at_lower_bound = cost_lower;
                cost_at_upper_bound = cost_upper;
            },
            _ => {panic!("Can not check for line search completion: values at bounds not known!")}
        }

        let search_line_gradient = 
            (cost_at_upper_bound - cost_at_lower_bound) / (2.*self.search_line.length());
        search_line_gradient <= self.optimization_settings.line_search_threshold
                    
    }

    pub fn quadratic_interpolation_iterate(&mut self) -> &mut Self {

        let cost_lower_bound = self.cost_lower_bound.unwrap();
        let cost_upper_bound = self.cost_upper_bound.unwrap();

        let central_point: f64;
        let cost_central_point: f64;

        match (self.central_point, self.cost_central_point) {
            (Option::Some(location), Option::Some(cost)) => {
                central_point = location;
                cost_central_point = cost;
            }
            _ => {
                self.central_point = Option::Some(0.5);
                central_point = 0.5;
                cost_central_point = self.cost(0.5);
            }
        }
        
        // fitted to cost = c_1 + c_2 x + c_3 x^2
        let c_1 = cost_lower_bound;
        let c_3 = 
            (cost_central_point - c_1) / (central_point * (central_point - 1.)) 
            + (c_1 - cost_upper_bound) / (central_point - 1.);
        let c_2 = 
            cost_upper_bound - cost_lower_bound - c_3;
        
        let lowest_point = - c_2 / (2. * c_3);
        let cost_lowest_point = self.cost(lowest_point);
        #[derive(Clone)]
        enum CostIs {
            LowerThenCentral,
            HigherThenCentral
        }

        enum LowestPoint {
            LeftOfCentral,
            RightOfCentral
        }

        let position_lowest: LowestPoint;
        let comparitive_cost: CostIs;
        if lowest_point < central_point {
            position_lowest = LowestPoint::LeftOfCentral;
        } else {
            position_lowest = LowestPoint::RightOfCentral;
        }

        if cost_lowest_point < cost_central_point {
            comparitive_cost = CostIs::LowerThenCentral;
        } else {
            comparitive_cost = CostIs::HigherThenCentral;
        }

        match (position_lowest, comparitive_cost.clone()) {
            (LowestPoint::LeftOfCentral, CostIs::LowerThenCentral) => {
                let length_after_transformation = central_point;
                let length_to_new_central_point = lowest_point / length_after_transformation;
                self.central_point = Option::Some(length_to_new_central_point);
                self.cost_central_point = Option::Some(cost_lowest_point);
                self.upper_bound_scale = Option::Some(1.);
                self.cost_upper_bound = Option::Some(cost_central_point);
                self.search_line.scale(central_point);
            },
            (LowestPoint::LeftOfCentral, CostIs::HigherThenCentral) => {
                let length_after_transformation = 1. -  lowest_point;
                let length_to_new_central_point = (central_point - lowest_point) / length_after_transformation;
                let scaling_factor_after_transform = (1. - lowest_point) / lowest_point;
                self.central_point = Option::Some(length_to_new_central_point);
                self.cost_central_point = Option::Some(cost_central_point);
                self.lower_bound_scale = Option::Some(0.);
                self.cost_lower_bound = Option::Some(cost_lowest_point);
                self.search_line.scale(lowest_point).move_base_to_end_position().scale(scaling_factor_after_transform);
            }, 
            (LowestPoint::RightOfCentral, CostIs::LowerThenCentral) => {
                let length_after_transformation = 1. - central_point;
                let length_to_new_central_point = (lowest_point - central_point) / length_after_transformation;
                let scaling_factor_after_transform = (1. - central_point) / central_point;
                self.central_point = Option::Some(length_to_new_central_point);
                self.cost_central_point = Option::Some(cost_lowest_point);
                self.lower_bound_scale = Option::Some(0.);
                self.cost_lower_bound = Option::Some(cost_central_point);
                self.search_line.scale(central_point);
                self.search_line.move_base_to_end_position();
                self.search_line.scale(scaling_factor_after_transform);
                // println!("change lower bound")
            },
            (LowestPoint::RightOfCentral, CostIs::HigherThenCentral) => {
                let length_after_transformation = lowest_point;
                let length_to_new_central_point = central_point / length_after_transformation;
                self.central_point = Option::Some(length_to_new_central_point);
                self.cost_central_point = Option::Some(cost_central_point);
                self.upper_bound_scale = Option::Some(1.);
                self.cost_upper_bound = Option::Some(cost_lowest_point);
                self.search_line.scale(lowest_point);
            }, 
        }
        
        // println!("central_point {}, lowest_point {}, is higher then central {}", central_point, lowest_point, stuck_condition_3);
        let stuck_condition_1 = central_point > 0.8;
        let stuck_condition_2 = lowest_point > 0.9;

        if stuck_condition_1 && stuck_condition_2 {
            self.cost_central_point = Option::None;
            self.central_point = Option::None;
        }

        let stuck_condition_1 = central_point < 0.2;
        let stuck_condition_2 = lowest_point < 0.1;

        if stuck_condition_1 && stuck_condition_2 {
            self.cost_central_point = Option::None;
            self.central_point = Option::None;
        }
        self
    }
}