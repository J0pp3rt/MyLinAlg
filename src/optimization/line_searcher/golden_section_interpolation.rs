use crate::{*};

impl LineSearcher {
    pub fn golden_section_interpolation_iterate_untill_convergence(&mut self) -> &mut Self {
        println!("using {}", self.optimization_settings.line_search_threshold );
        while self.golden_ratio_is_completed().not() {
            self.golden_section_interpolation_iterate();
        }
        self
    }

    fn golden_ratio_is_completed(&self) -> bool {
        let coord_distance = self.search_line.length();
        coord_distance < self.optimization_settings.line_search_threshold 
    }

    pub fn golden_section_interpolation_iterate(&mut self) -> &mut Self {

        let inner_point_low: f64;
        let cost_inner_point_low: f64;
        match (self.inner_point_low, self.cost_inner_point_low) {
            (Option::Some(_), Option::Some(cost)) => {
                cost_inner_point_low = cost;
            }
            _ => {
                inner_point_low = 1. - golden_ratio_conjugate();
                cost_inner_point_low = self.cost(inner_point_low);
            }
        }

        let inner_point_high: f64;
        let cost_inner_point_high: f64;
        match (self.inner_point_high, self.cost_inner_point_high) {
            (Option::Some(_), Option::Some(cost)) => {
                cost_inner_point_high = cost;
            }
            _ => {
                inner_point_high = golden_ratio_conjugate();
                cost_inner_point_high = self.cost(inner_point_high);
            }
        }

        enum CostHigherOn {
            InnerPointLow,
            InnerPointHigh,
        }

        let highest_value_on: CostHigherOn;
        if cost_inner_point_low > cost_inner_point_high {
            highest_value_on = CostHigherOn::InnerPointLow;
        } else {
            highest_value_on = CostHigherOn::InnerPointHigh;
        }

        let lenght_after_transforamtion = self.search_line.length() * golden_ratio_conjugate();

        match highest_value_on {
            CostHigherOn::InnerPointLow => {
                self.search_line
                    .scale(1.-golden_ratio_conjugate() )
                    .move_base_to_end_position()
                    .scale_to_length(lenght_after_transforamtion);
                self.lower_bound_scale = Option::Some(0.);
                self.cost_lower_bound = Option::Some(cost_inner_point_low);
                self.inner_point_low = Option::Some(1. - golden_ratio_conjugate());
                self.cost_inner_point_low = Option::Some(cost_inner_point_high);
                self.inner_point_high = Option::None; 
                self.cost_inner_point_high = Option::None;
            },
            CostHigherOn::InnerPointHigh => {
                self.search_line
                    .scale_to_length(lenght_after_transforamtion);
                self.upper_bound_scale = Option::Some(1.);
                self.cost_upper_bound = Option::Some(cost_inner_point_high);
                self.inner_point_low = Option::None;
                self.cost_inner_point_low = Option::None;
                self.inner_point_high = Option::Some(golden_ratio_conjugate());
                self.cost_inner_point_high = Option::Some(cost_inner_point_low);
            },
        }
        self
    }
}