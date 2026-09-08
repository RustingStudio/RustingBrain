//! Everything that turns a `Row` into the exact vector the network expects -
//! and that must be saved alongside the weights.

use crate::data::{Row, PLANS};
use std::error::Error;
use std::fs;

pub struct Preprocessing {
    /// Fills in blank `monthly_hours` cells. Learned from the training set only.
    pub hours_median: f32,
    pub min: Vec<f32>,
    pub max: Vec<f32>,
}

impl Preprocessing {
    /// Fit on the TRAINING rows only. Anything learned from validation or test
    /// data here is leakage (chapter 4).
    pub fn fit(rows: &[Row]) -> Self {
        let mut hours: Vec<f32> = rows.iter().filter_map(|r| r.monthly_hours).collect();
        hours.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let hours_median = hours[hours.len() / 2];

        let encoded: Vec<Vec<f32>> = rows
            .iter()
            .map(|r| encode(r, hours_median))
            .collect();
        let w = encoded[0].len();
        let mut min = vec![f32::INFINITY; w];
        let mut max = vec![f32::NEG_INFINITY; w];
        for row in &encoded {
            for (i, &v) in row.iter().enumerate() {
                min[i] = min[i].min(v);
                max[i] = max[i].max(v);
            }
        }
        Self { hours_median, min, max }
    }

    /// Row -> the 6 numbers the network sees. The only path into `predict`.
    pub fn transform(&self, row: &Row) -> Vec<f32> {
        let raw = encode(row, self.hours_median);
        raw.iter()
            .enumerate()
            .map(|(i, &v)| {
                let s = self.max[i] - self.min[i];
                if s.abs() < f32::EPSILON { 0.0 } else { (v - self.min[i]) / s }
            })
            .collect()
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let join = |v: &[f32]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        let out = format!(
            "version 1\nhours_median {}\nmin {}\nmax {}\n",
            self.hours_median,
            join(&self.min),
            join(&self.max)
        );
        fs::write(path, out)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let text = fs::read_to_string(path)?;
        let (mut hours_median, mut min, mut max) = (None, Vec::new(), Vec::new());
        for line in text.lines() {
            let (key, value) = line.split_once(' ').ok_or("malformed line")?;
            match key {
                "version" if value.trim() != "1" => {
                    return Err(format!("unsupported preprocessing version {value}").into())
                }
                "version" => {}
                "hours_median" => hours_median = Some(value.trim().parse()?),
                "min" => min = value.split(',').map(|v| v.parse()).collect::<Result<_, _>>()?,
                "max" => max = value.split(',').map(|v| v.parse()).collect::<Result<_, _>>()?,
                other => return Err(format!("unknown key {other}").into()),
            }
        }
        let hours_median = hours_median.ok_or("missing hours_median")?;
        if min.len() != max.len() || min.is_empty() {
            return Err("incomplete preprocessing file".into());
        }
        Ok(Self { hours_median, min, max })
    }
}

/// Numbers stay as they are; `plan` becomes three one-hot columns.
fn encode(r: &Row, hours_median: f32) -> Vec<f32> {
    let mut v = vec![
        r.months_active,
        r.monthly_hours.unwrap_or(hours_median),
        r.support_tickets,
    ];
    v.extend(PLANS.iter().map(|p| if *p == r.plan { 1.0 } else { 0.0 }));
    v
}
