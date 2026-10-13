//! `POST /v1/systemone`: System One decision answers over HTTP, in each
//! model family's own wire shape.

pub mod answer;
pub mod error;
pub mod handler;
pub mod kev_answer;
pub mod response;

pub use answer::{answers_json, py_round};
pub use error::failure_response;
pub use handler::systemone;
pub use kev_answer::kev_answer;
pub use response::{kev_response_body, response_body, KevUsage, ResponseFacts};
