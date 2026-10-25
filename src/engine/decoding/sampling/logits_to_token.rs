//! Fused logits-to-token sampling: the `SamplingOps` kernel call over logits
//! that `prepare` has shaped on the host.

use anyhow::{anyhow, Result};

use boostr::inference::sampling::DryParams;
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, DeviceGrammarDfa, GrammarDfaOps, NormalizationOps,
    Runtime, SampleSeed, SamplingOps, SamplingScratch, ScalarOps, Tensor, TypeConversionOps,
    UnaryOps,
};

use crate::config::GenerationConfig;

use crate::engine::executor::Executor;

impl<R: Runtime<DType = DType>> Executor<R>
where
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + NormalizationOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + TypeConversionOps<R>
        + SamplingOps<R>
        + GrammarDfaOps<R>
        + ModelClient<R>
        + boostr::quant::traits::DequantOps<R>,
{
    /// Fused logits-to-token: narrow + cast + penalties + argmax/sample in a single kernel launch.
    ///
    /// Returns `[1]` I64 tensor on device. All decode paths (greedy, greedy+penalties,
    /// non-greedy) return the same type, enabling uniform pipelined decode.
    /// With `gpu_grammar`, the device-side grammar DFA masks the logits first.
    /// `dry` is the request's DRY penalty, its breakers resolved once at
    /// the start of the request. `None` applies no DRY penalty.
    /// `seed` names this step's Philox draw. `None` samples unseeded.
    pub(crate) fn logits_to_token_on_device_with_grammar(
        &self,
        logits: &Tensor<R>,
        recent_tokens: &[u32],
        gen_config: &GenerationConfig,
        dry: Option<&DryParams>,
        seed: Option<SampleSeed>,
        gpu_grammar: Option<&DeviceGrammarDfa<R>>,
    ) -> Result<Tensor<R>> {
        let ready =
            self.prepare_logits_to_token(logits, recent_tokens, gen_config, dry, gpu_grammar)?;
        let client = R::default_client(self.device());
        client
            .logits_to_token(
                &ready.logits,
                &ready.ids_tensor,
                &ready.cnts_tensor,
                ready.num_unique,
                gen_config.repeat_penalty,
                gen_config.frequency_penalty,
                gen_config.presence_penalty,
                ready.temperature,
                gen_config.top_k,
                gen_config.top_p,
                gen_config.min_p,
                seed,
            )
            .map_err(|e| anyhow!("logits_to_token failed: {}", e))
    }

    /// Like [`Self::logits_to_token_on_device_with_grammar`], but samples
    /// through [`SamplingOps::logits_to_token_into`], writing the token into
    /// `scratch.output` instead of allocating.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn logits_to_token_on_device_with_grammar_into(
        &self,
        logits: &Tensor<R>,
        recent_tokens: &[u32],
        gen_config: &GenerationConfig,
        dry: Option<&DryParams>,
        seed: Option<SampleSeed>,
        gpu_grammar: Option<&DeviceGrammarDfa<R>>,
        scratch: &SamplingScratch<R>,
    ) -> Result<Tensor<R>> {
        let ready =
            self.prepare_logits_to_token(logits, recent_tokens, gen_config, dry, gpu_grammar)?;
        let client = R::default_client(self.device());
        client
            .logits_to_token_into(
                &ready.logits,
                &ready.ids_tensor,
                &ready.cnts_tensor,
                ready.num_unique,
                gen_config.repeat_penalty,
                gen_config.frequency_penalty,
                gen_config.presence_penalty,
                ready.temperature,
                gen_config.top_k,
                gen_config.top_p,
                gen_config.min_p,
                seed,
                scratch,
            )
            .map_err(|e| anyhow!("logits_to_token_into failed: {}", e))?;
        Ok(scratch.output.clone())
    }
}

#[cfg(test)]
mod tests {
    //! Golden outputs of the host-side logit shaping. The expected values
    //! are bit patterns, so any change in operation order fails the test.

    use super::*;
    use std::collections::HashMap;

    use boostr::CpuRuntime;

    use boostr::inference::sampling::{
        apply_dry_penalty, apply_typical_filter, DynaTemp, MirostatV2,
    };

    use super::super::dry_params;
    use crate::engine::test_support::tiny_executor;

    // The four functions below are the only lines that name the
    // implementation under test.

    fn dry(logits: &mut [f32], history: &[u32], gen: &GenerationConfig) -> Result<()> {
        let tokenizer = crate::tokenizer::from_pretrained("llama3")?;
        let params = dry_params(&tokenizer, gen)
            .map_err(|e| anyhow!(e))?
            .ok_or_else(|| anyhow!("DRY is off"))?;
        apply_dry_penalty(logits, history, &params);
        Ok(())
    }

    fn typical(logits: &mut [f32], typical_p: f32) {
        apply_typical_filter(logits, typical_p);
    }

    fn dynatemp(row: &[f32], gen: &GenerationConfig) -> Result<f32> {
        Ok(DynaTemp::from(gen).temperature(row))
    }

    /// `(token, logprob bits)` of each step, one logits row per step.
    fn mirostat(rows: &[Vec<f32>], temperature: f32) -> Result<Vec<(u32, u32)>> {
        let mut state = MirostatV2::new(5.0, 0.1, Some(42));
        rows.iter()
            .map(|row| {
                let (id, lp) = state.sample(row, temperature)?;
                Ok((id, lp.to_bits()))
            })
            .collect()
    }

    /// Logits on a 0.25 grid in `[-4, 4]`, so many values tie.
    fn logits(vocab: usize, salt: u64) -> Vec<f32> {
        let mut s = salt.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..vocab)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) % 33) as f32 * 0.25 - 4.0
            })
            .collect()
    }

    /// Repeated n-grams, one id past the small vocab, and a tail that
    /// recurs earlier in the history.
    const HISTORY: [u32; 24] = [
        5, 9, 2, 5, 9, 2, 7, 1, 5, 9, 3, 3, 3, 8, 1, 5, 9, 2, 40, 7, 1, 5, 9, 2,
    ];

    /// FNV-1a over the bit patterns of `values`.
    fn digest(values: impl IntoIterator<Item = u32>) -> u64 {
        values.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, v| {
            v.to_le_bytes()
                .iter()
                .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
        })
    }

    fn bits(values: &[f32]) -> u64 {
        digest(values.iter().map(|v| v.to_bits()))
    }

    fn dry_config(multiplier: f32, base: usize, allowed_length: usize) -> GenerationConfig {
        GenerationConfig {
            dry_multiplier: multiplier,
            dry_base: base,
            dry_allowed_length: allowed_length,
            ..GenerationConfig::default()
        }
    }

    /// The token ids `logits_to_token_on_device_with_grammar` samples over
    /// 24 seeded steps of `gen`, each step's id appended to the history.
    fn device_tokens(gen: &GenerationConfig) -> Result<Vec<i64>> {
        let exec = tiny_executor()?;
        let vocab = exec.vocab_size();
        let dry = dry_params(exec.tokenizer(), gen).map_err(|e| anyhow!(e))?;
        let mut history = vec![5, 9, 2, 5, 9];
        let mut out = Vec::new();
        for step in 0..24u64 {
            let row = logits(vocab, 100 + step);
            let t = Tensor::<CpuRuntime>::from_slice(&row, &[1, 1, vocab], exec.device())?;
            let seed = Some(SampleSeed::new(11, step));
            let token = exec.logits_to_token_on_device_with_grammar(
                &t,
                &history,
                gen,
                dry.as_ref(),
                seed,
                None,
            )?;
            let id = token.to_vec::<i64>()[0];
            history.push(id as u32);
            out.push(id);
        }
        Ok(out)
    }

    fn dry_sampling() -> GenerationConfig {
        GenerationConfig {
            temperature: 0.9,
            dry_multiplier: 0.8,
            repeat_penalty: 1.1,
            repeat_last_n: 8,
            seed: Some(11),
            ..GenerationConfig::default()
        }
    }

    fn every_host_filter() -> GenerationConfig {
        GenerationConfig {
            typical_p: 0.95,
            dynatemp_range: 0.3,
            frequency_penalty: 0.2,
            presence_penalty: 0.1,
            logit_bias: HashMap::from([(3, 1.5), (11, -2.0), (999, 4.0)]),
            ..dry_sampling()
        }
    }

    #[test]
    fn dry_penalty_matches_the_golden_bits() -> Result<()> {
        let cases: [(usize, f32, usize, usize, u64); 10] = [
            (32, 0.8, 2, 0, 0xa64f_eaef_d3c1_d3c1),
            (32, 1.5, 3, 0, 0x4028_28d5_8b23_f2fb),
            (32, 0.8, 2, 10, 0x5542_b013_8626_147b),
            (32, 0.8, 2, 3, 0x5542_b013_8626_147b),
            (32, 0.8, 30, 0, 0x5542_b013_8626_147b),
            (1000, 0.8, 2, 0, 0x6708_f404_f542_c94c),
            (1000, 1.5, 3, 0, 0x69ee_7f31_2b0f_25b0),
            (1000, 0.8, 2, 10, 0x72c9_9178_229c_6a62),
            (1000, 0.8, 2, 3, 0xd463_e88c_9cbb_f4b5),
            (1000, 0.8, 30, 0, 0xd463_e88c_9cbb_f4b5),
        ];
        for (vocab, multiplier, base, allowed, want) in cases {
            let mut l = logits(vocab, 1);
            dry(&mut l, &HISTORY, &dry_config(multiplier, base, allowed))?;
            assert_eq!(
                bits(&l),
                want,
                "vocab {vocab} dry {multiplier} {base} {allowed}"
            );
        }
        assert_eq!(
            bits(&logits(32, 1)),
            0x5542_b013_8626_147b,
            "unpenalized row"
        );
        Ok(())
    }

    #[test]
    fn typical_filter_matches_the_golden_bits() {
        let cases: [(usize, f32, u64, usize); 8] = [
            (32, 0.2, 0xc332_2f45_5512_4858, 3),
            (32, 0.5, 0xfc9c_2906_e8c5_feb8, 7),
            (32, 0.9, 0x4262_889f_6a42_5545, 11),
            (32, 1.0, 0x4a24_1f4c_a630_9ac2, 32),
            (1000, 0.2, 0x7e99_71a2_4133_3a58, 69),
            (1000, 0.5, 0x81b8_9b9d_faaf_2445, 170),
            (1000, 0.9, 0xf32b_374b_c203_dcf5, 317),
            (1000, 1.0, 0x700c_5c14_5d82_5632, 1000),
        ];
        for (vocab, p, want, want_kept) in cases {
            let mut l = logits(vocab, 2);
            typical(&mut l, p);
            let kept = l.iter().filter(|v| v.is_finite()).count();
            assert_eq!(
                (bits(&l), kept),
                (want, want_kept),
                "vocab {vocab} typical {p}"
            );
        }
    }

    #[test]
    fn dynamic_temperature_matches_the_golden_bits() -> Result<()> {
        let cases: [(usize, f32, u32); 6] = [
            (32, 1.0, 0x3f62_8af4),
            (32, 2.0, 0x3f24_63b4),
            (32, 0.5, 0x3f88_4ba9),
            (1000, 1.0, 0x3f92_fc2c),
            (1000, 2.0, 0x3f82_83d4),
            (1000, 0.5, 0x3f9c_4b26),
        ];
        for (vocab, exponent, want) in cases {
            let gen = GenerationConfig {
                temperature: 0.8,
                dynatemp_range: 0.5,
                dynatemp_exponent: exponent,
                ..GenerationConfig::default()
            };
            let got = dynatemp(&logits(vocab, 3), &gen)?.to_bits();
            assert_eq!(got, want, "vocab {vocab} exponent {exponent}");
        }
        let mut one_hot = vec![-1e4f32; 32];
        one_hot[7] = 10.0;
        let gen = GenerationConfig {
            temperature: 0.8,
            dynatemp_range: 0.5,
            ..GenerationConfig::default()
        };
        assert_eq!(dynatemp(&one_hot, &gen)?.to_bits(), 0x3e99_999a, "one-hot");
        Ok(())
    }

    #[test]
    fn mirostat_matches_the_golden_stream() -> Result<()> {
        let cases: [(usize, f32, [u32; 50], u64); 4] = [
            (
                32,
                0.8,
                [
                    13, 13, 21, 31, 12, 19, 9, 29, 11, 28, 3, 11, 12, 24, 2, 11, 20, 5, 26, 7, 15,
                    2, 21, 6, 3, 24, 11, 15, 5, 4, 15, 26, 27, 1, 12, 26, 20, 1, 18, 6, 10, 24, 12,
                    26, 23, 9, 0, 7, 28, 4,
                ],
                0x553c_6901_6ebe_9b2d,
            ),
            (
                32,
                1.0,
                [
                    13, 13, 21, 18, 17, 24, 16, 1, 29, 28, 3, 2, 27, 21, 2, 21, 28, 5, 26, 7, 30,
                    28, 27, 25, 4, 14, 16, 15, 25, 4, 21, 22, 16, 1, 12, 26, 5, 1, 7, 6, 23, 4, 12,
                    26, 30, 23, 7, 7, 28, 4,
                ],
                0x1d51_b3cf_a9c8_1ab1,
            ),
            (
                1000,
                0.8,
                [
                    445, 118, 922, 817, 652, 265, 928, 978, 702, 131, 876, 731, 410, 835, 72, 17,
                    15, 5, 17, 77, 124, 109, 18, 2, 85, 69, 3, 240, 63, 32, 30, 59, 84, 1, 65, 87,
                    50, 32, 17, 17, 10, 30, 78, 5, 7, 13, 0, 7, 28, 13,
                ],
                0x05da_12c5_698f_15a5,
            ),
            (
                1000,
                1.0,
                [
                    494, 372, 94, 56, 434, 310, 928, 1, 836, 81, 582, 536, 4, 18, 72, 17, 15, 5,
                    17, 77, 124, 109, 18, 2, 85, 69, 3, 240, 63, 32, 30, 59, 84, 1, 65, 87, 50, 32,
                    17, 17, 10, 30, 78, 5, 7, 13, 0, 7, 28, 13,
                ],
                0x74b5_180d_8818_ac7b,
            ),
        ];
        for (vocab, temperature, want_ids, want_logprobs) in cases {
            let rows: Vec<Vec<f32>> = (0..50).map(|s| logits(vocab, 200 + s)).collect();
            let run = mirostat(&rows, temperature)?;
            let ids: Vec<u32> = run.iter().map(|r| r.0).collect();
            assert_eq!(ids, want_ids, "vocab {vocab} temperature {temperature}");
            let logprobs = digest(run.iter().map(|r| r.1));
            assert_eq!(
                logprobs, want_logprobs,
                "vocab {vocab} temperature {temperature}"
            );
        }
        Ok(())
    }

    #[test]
    fn logits_to_token_on_device_with_grammar_matches_the_golden_tokens() -> Result<()> {
        assert_eq!(
            device_tokens(&dry_sampling())?,
            [
                29, 21, 18, 17, 0, 23, 1, 22, 30, 27, 22, 2, 8, 7, 24, 23, 4, 28, 15, 12, 15, 26,
                8, 24
            ]
        );
        assert_eq!(
            device_tokens(&every_host_filter())?,
            [
                29, 21, 18, 17, 0, 23, 1, 22, 30, 27, 25, 2, 3, 7, 24, 23, 3, 28, 15, 8, 15, 26, 7,
                24
            ]
        );
        Ok(())
    }
}
