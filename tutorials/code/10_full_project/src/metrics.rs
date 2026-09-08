//! Counting mistakes properly, and choosing the threshold.

pub struct Confusion {
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub fn_: usize,
}

impl Confusion {
    pub fn count(scores: &[f32], truth: &[bool], threshold: f32) -> Self {
        let (mut tp, mut fp, mut tn, mut fn_) = (0, 0, 0, 0);
        for (&s, &t) in scores.iter().zip(truth) {
            match (s >= threshold, t) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, false) => tn += 1,
                (false, true) => fn_ += 1,
            }
        }
        Self { tp, fp, tn, fn_ }
    }

    pub fn accuracy(&self) -> f32 {
        (self.tp + self.tn) as f32 / (self.tp + self.fp + self.tn + self.fn_) as f32
    }
    pub fn precision(&self) -> f32 {
        if self.tp + self.fp == 0 { 0.0 } else { self.tp as f32 / (self.tp + self.fp) as f32 }
    }
    pub fn recall(&self) -> f32 {
        if self.tp + self.fn_ == 0 { 0.0 } else { self.tp as f32 / (self.tp + self.fn_) as f32 }
    }
    pub fn f1(&self) -> f32 {
        let (p, r) = (self.precision(), self.recall());
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }

    pub fn print(&self, label: &str) {
        println!("  {label}");
        println!("                 predicted stay   predicted churn");
        println!("  actual stay    {:>14}   {:>15}", self.tn, self.fp);
        println!("  actual churn   {:>14}   {:>15}", self.fn_, self.tp);
        println!(
            "  accuracy {:.1}%   precision {:.1}%   recall {:.1}%   F1 {:.3}",
            self.accuracy() * 100.0,
            self.precision() * 100.0,
            self.recall() * 100.0,
            self.f1()
        );
    }
}
