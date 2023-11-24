

#[derive(Clone)]
pub enum MarkerStyle {
    None,
    Circle(u32),
    Diamand(u32),
    Cross(u32),
}

#[derive(Clone)]
pub enum MarkerFill {
    Filled,
    NotFilled
}

#[derive(Clone)]
pub enum LineStyle {
    Solid,
    Dashed,
}

#[derive(PartialEq, Clone)]
pub enum PlotAxisScaling {
    Linear,
    Log,
    NoAxis
}