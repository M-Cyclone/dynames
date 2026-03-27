pub fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}
