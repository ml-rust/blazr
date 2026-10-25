//! Stop-sequence cutting for one executor stream, on its decode thread.

pub(super) use super::stop_forwarder_state::StopForwarder;
use super::streaming::StreamToken;
use crate::engine::decode_stream::decode_output::{DecodeOutput, Emit, StreamEnd};
use crate::engine::{FinishReason, GeneratedToken};

impl DecodeOutput for StopForwarder {
    type Item = StreamToken;

    fn token(&mut self, token: anyhow::Result<GeneratedToken>) -> Emit<StreamToken> {
        let token = match token {
            Ok(token) => token,
            Err(e) => {
                tracing::error!("{} error during streaming: {}", self.label, e);
                self.errored = true;
                return Emit::Last(StreamToken::error(e.to_string()));
            }
        };
        if let Some(metrics) = self.metrics.as_mut() {
            metrics.token();
        }

        // A match ends with `Stop`. The last token carries the held-back
        // text, which is not a stop.
        let cut = self.matcher.cut_token(&token.text, token.finish_reason);
        let held = cut.text.is_empty() && !token.text.is_empty();
        let item = StreamToken::text(cut.text, cut.finish).with_logprobs(&token, self.logprobs);
        if cut.finish.is_some() {
            return Emit::Last(item.with_usage(self.usage()));
        }
        if held && item.logprob.is_none() {
            return Emit::Skip;
        }
        Emit::Send(item)
    }

    fn end(&mut self, why: StreamEnd) -> Option<StreamToken> {
        // The stream ended without a final token: flush the held-back text.
        let close = (why == StreamEnd::Drained).then(|| {
            let tail = self.matcher.finish();
            StreamToken::text(tail, Some(FinishReason::Length)).with_usage(self.usage())
        });
        if let Some(metrics) = self.metrics.take() {
            if !self.errored {
                metrics.finish();
            }
        }
        close
    }

    fn failure(message: String) -> StreamToken {
        StreamToken::error(message)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::stream_usage::StreamUsage;
    use super::*;
    use crate::config::GenerationConfig;

    /// Forward `stream` through a [`StopForwarder`] to `tx` on this task.
    async fn forward_with_stop_sequences<S>(
        stream: S,
        prompt_len: usize,
        gen_config: &GenerationConfig,
        label: &'static str,
        tx: tokio::sync::mpsc::Sender<StreamToken>,
    ) where
        S: futures::Stream<Item = anyhow::Result<GeneratedToken>>,
    {
        let mut out = StopForwarder::new(prompt_len, gen_config, label);
        crate::engine::decode_stream::decode_output::forward_tokens(
            stream,
            &mut out,
            &tx,
            None,
            |_| {},
        )
        .await;
    }

    fn tok(
        text: &str,
        finish_reason: Option<FinishReason>,
    ) -> anyhow::Result<crate::engine::GeneratedToken> {
        Ok(crate::engine::GeneratedToken {
            token_id: 0,
            text: text.to_string(),
            logprob: None,
            top_logprobs: None,
            finish_reason,
        })
    }

    /// Run the forwarder over `tokens` and collect every item it sends.
    async fn forward_items(
        tokens: Vec<anyhow::Result<crate::engine::GeneratedToken>>,
        stops: &[&str],
        logprobs: bool,
    ) -> Vec<StreamToken> {
        let gen_config = GenerationConfig {
            stop_sequences: stops.iter().map(|s| s.to_string()).collect(),
            logprobs,
            ..GenerationConfig::default()
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        let stream = futures::stream::iter(tokens);
        forward_with_stop_sequences(stream, 3, &gen_config, "Test", tx).await;
        let mut out = Vec::new();
        while let Some(t) = rx.recv().await {
            out.push(t);
        }
        out
    }

    /// Run the forwarder over `tokens` and collect what the client receives.
    async fn forward(
        tokens: Vec<anyhow::Result<crate::engine::GeneratedToken>>,
        stops: &[&str],
    ) -> Vec<(String, Option<FinishReason>)> {
        forward_items(tokens, stops, false)
            .await
            .into_iter()
            .map(|t| (t.text, t.finish_reason))
            .collect()
    }

    fn with_logprob(id: u32, text: &str) -> anyhow::Result<crate::engine::GeneratedToken> {
        let mut token = tok(text, None)?;
        token.token_id = id;
        token.logprob = Some(-(id as f32));
        Ok(token)
    }

    /// `E` and `N` start the stop `END`, so the matcher holds their text.
    /// Their items still go out, empty, each with its own logprob.
    #[tokio::test]
    async fn held_back_tokens_send_their_logprobs_when_asked() {
        let tokens = || {
            vec![
                with_logprob(1, "a"),
                with_logprob(2, "E"),
                with_logprob(3, "N"),
                with_logprob(4, "x"),
            ]
        };
        let out = forward_items(tokens(), &["END"], true).await;
        let got: Vec<(String, Option<u32>, Option<f32>)> = out
            .iter()
            .map(|t| {
                let lp = t.logprob.as_ref();
                (
                    t.text.clone(),
                    lp.map(|l| l.token_id),
                    lp.map(|l| l.logprob),
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                ("a".to_string(), Some(1), Some(-1.0)),
                (String::new(), Some(2), Some(-2.0)),
                (String::new(), Some(3), Some(-3.0)),
                ("ENx".to_string(), Some(4), Some(-4.0)),
                (String::new(), None, None),
            ]
        );
        assert_eq!(
            out.last().and_then(|t| t.finish_reason),
            Some(FinishReason::Length)
        );

        let plain = forward_items(tokens(), &["END"], false).await;
        assert!(plain
            .iter()
            .all(|t| t.logprob.is_none() && t.top_logprobs.is_none()));
        let texts: Vec<&str> = plain.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            texts,
            ["a", "ENx", ""],
            "held tokens send nothing without logprobs"
        );
    }

    /// The token counts of each item, in order.
    async fn usages(
        tokens: Vec<anyhow::Result<crate::engine::GeneratedToken>>,
        stops: &[&str],
    ) -> Vec<Option<StreamUsage>> {
        forward_items(tokens, stops, false)
            .await
            .into_iter()
            .map(|t| t.usage)
            .collect()
    }

    /// Only the final item carries counts. Held-back and stop tokens count.
    #[tokio::test]
    async fn the_final_item_carries_every_generated_token() {
        let counts = StreamUsage::new(3, 3);
        let stop = usages(
            vec![tok("hi E", None), tok("N", None), tok("D!", None)],
            &["END"],
        )
        .await;
        assert_eq!(stop, vec![None, Some(counts)], "stop sequence");

        let eos = usages(
            vec![
                tok("a", None),
                tok("b", None),
                tok("c", Some(FinishReason::Eos)),
            ],
            &[],
        )
        .await;
        assert_eq!(eos, vec![None, None, Some(counts)], "engine finish");

        let ended = usages(vec![tok("x EN", None), tok("D", None)], &["ENDX"]).await;
        assert_eq!(
            ended,
            vec![None, Some(StreamUsage::new(3, 2))],
            "a stream end with held text closes with the counts"
        );
    }

    #[tokio::test]
    async fn split_stop_never_reaches_the_client() {
        let out = forward(
            vec![tok("hi E", None), tok("N", None), tok("D!", None)],
            &["END"],
        )
        .await;
        let text: String = out.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(text, "hi ");
        assert_eq!(out.last().and_then(|(_, r)| *r), Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn no_stops_forward_every_token_unchanged() {
        let out = forward(
            vec![
                tok("a", None),
                tok("", None),
                tok("b", Some(FinishReason::Eos)),
            ],
            &[],
        )
        .await;
        assert_eq!(
            out,
            vec![
                ("a".to_string(), None),
                (String::new(), None),
                ("b".to_string(), Some(FinishReason::Eos)),
            ]
        );
    }

    #[tokio::test]
    async fn final_token_flushes_held_back_text() {
        let out = forward(
            vec![tok("x EN", None), tok("", Some(FinishReason::Length))],
            &["END"],
        )
        .await;
        assert_eq!(
            out,
            vec![
                ("x ".to_string(), None),
                ("EN".to_string(), Some(FinishReason::Length)),
            ]
        );
    }

    #[tokio::test]
    async fn stream_end_without_final_token_flushes_held_back_text() {
        let out = forward(vec![tok("x EN", None)], &["END"]).await;
        let text: String = out.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(text, "x EN");
        assert_eq!(out.last().and_then(|(_, r)| *r), Some(FinishReason::Length));
    }

    type CpuExecutor = crate::engine::Executor<boostr::CpuRuntime>;

    /// What the client receives when `exec` streams `prompt` through the forwarder.
    async fn forward_engine(
        exec: &CpuExecutor,
        prompt: &[u32],
        cfg: &GenerationConfig,
    ) -> Vec<(String, Option<FinishReason>)> {
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        forward_with_stop_sequences(exec.generate(prompt, cfg), prompt.len(), cfg, "Test", tx)
            .await;
        let mut out = Vec::new();
        while let Some(t) = rx.recv().await {
            out.push((t.text, t.finish_reason));
        }
        out
    }

    #[tokio::test]
    async fn speculative_streams_match_plain_streams() -> anyhow::Result<()> {
        use crate::config::SpeculativeDecodingConfig;
        use crate::engine::batch_test_rig::greedy;
        use crate::engine::test_support::seeded_executor_ctx;

        let plain = seeded_executor_ctx(0x5bec, 64)?;
        let mut spec = seeded_executor_ctx(0x5bec, 64)?;
        spec.config.inference.speculative = Some(SpeculativeDecodingConfig {
            enabled: true,
            num_draft_tokens: Some(5),
            ..SpeculativeDecodingConfig::default()
        });
        let (plain, spec) = (Arc::new(plain), Arc::new(spec));
        let probe = GenerationConfig {
            logprobs: true,
            top_logprobs: 2,
            ..greedy(16)
        };
        let prompts: [&[u32]; 3] = [
            &[3, 7, 11, 3, 7, 11, 3, 7, 11],
            &[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3],
            &[4, 4, 4, 4, 4, 4],
        ];
        let mut compared = 0;
        for prompt in prompts {
            // Skip a prompt whose greedy run has a top-2 tie the two paths can break apart.
            let result = plain.generate_text(prompt, &probe).await?;
            let tokens = result.token_logprobs.unwrap_or_default();
            let tied = tokens.iter().any(|t| match t.top_logprobs.as_deref() {
                Some([a, b, ..]) => (a.logprob - b.logprob).abs() < 1e-4,
                _ => true,
            });
            if tied {
                continue;
            }
            // Every id under 32 is one ASCII character, so a byte range is a token range.
            let text: String = tokens.iter().map(|t| t.text.as_str()).collect();
            if text.len() != tokens.len() || !text.is_ascii() {
                return Err(anyhow::anyhow!(
                    "{text:?} is not one ASCII character per token"
                ));
            }
            let stop = GenerationConfig {
                stop_sequences: vec![text[3..5].to_string()],
                ..greedy(16)
            };
            for cfg in [greedy(16), stop] {
                let want = forward_engine(&plain, prompt, &cfg).await;
                assert_eq!(
                    forward_engine(&spec, prompt, &cfg).await,
                    want,
                    "{prompt:?}"
                );
            }
            compared += 1;
        }
        assert!(compared > 0, "every prompt has a near-tie");
        Ok(())
    }
}
