# Chapter 10 — churn prediction

    cargo new churn
    cd churn
    cargo add rusting_brain

Copy these `src/*.rs` files in, and copy `tutorials/data/subscriptions.csv`
to `churn/data/subscriptions.csv`. Then:

    cargo run --release -- train
    cargo run --release -- evaluate
    cargo run --release -- predict 3 45.0 6 basic
    cargo run --release -- predict 12 ? 3 plus      # ? = missing reading

`train` writes `churn_model.json`, `churn_model.prep` and
`churn_model.threshold` into the current directory.
