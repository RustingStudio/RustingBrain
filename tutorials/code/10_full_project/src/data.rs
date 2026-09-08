//! Reading the CSV and turning it into numbers.

use std::error::Error;
use std::fs;

/// One row of the raw file, still in its original units, with `monthly_hours`
/// optional because the file genuinely has blanks.
#[derive(Clone, Debug)]
pub struct Row {
    pub months_active: f32,
    pub monthly_hours: Option<f32>,
    pub support_tickets: f32,
    pub plan: String,
    pub cancelled: bool,
}

pub const PLANS: [&str; 3] = ["basic", "plus", "pro"];

/// Feature layout after encoding. Keeping this in one place means the training
/// code and the prediction code can never disagree about column order.
pub const FEATURE_NAMES: [&str; 6] = [
    "months_active",
    "monthly_hours",
    "support_tickets",
    "plan=basic",
    "plan=plus",
    "plan=pro",
];

pub fn read_csv(path: &str) -> Result<Vec<Row>, Box<dyn Error>> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("could not read {path}: {e}"))?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().ok_or("file is empty")?.split(',').collect();
    let expected = [
        "months_active",
        "monthly_hours",
        "support_tickets",
        "plan",
        "cancelled",
    ];
    if header != expected {
        return Err(format!("unexpected header: {header:?}").into());
    }

    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        let line_no = i + 2; // 1-based, and the header was line 1
        if line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        if c.len() != expected.len() {
            return Err(format!("line {line_no}: expected 5 columns, found {}", c.len()).into());
        }
        let num = |s: &str, name: &str| -> Result<f32, Box<dyn Error>> {
            s.trim()
                .parse::<f32>()
                .map_err(|_| format!("line {line_no}: {name} is not a number: {s:?}").into())
        };
        let hours = c[1].trim();
        let plan = c[3].trim().to_string();
        if !PLANS.contains(&plan.as_str()) {
            return Err(format!("line {line_no}: unknown plan {plan:?}").into());
        }
        let cancelled = match c[4].trim() {
            "yes" => true,
            "no" => false,
            other => return Err(format!("line {line_no}: cancelled must be yes/no, got {other:?}").into()),
        };
        rows.push(Row {
            months_active: num(c[0], "months_active")?,
            monthly_hours: if hours.is_empty() { None } else { Some(num(hours, "monthly_hours")?) },
            support_tickets: num(c[2], "support_tickets")?,
            plan,
            cancelled,
        });
    }
    if rows.is_empty() {
        return Err("no data rows".into());
    }
    Ok(rows)
}
