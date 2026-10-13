//! One base model load: from a model path to a started executor. Every
//! step reads the disk or uploads to the device, so the scheduler runs
//! [`LoadJob::run`] off the async executor.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, ScalarOps, UnaryOps,
};

use super::parts::{load_model_parts, start_executor};
use crate::config::{BlazrConfig, DtypeRequest, InferenceOverlay, VisionOverrides};
use crate::engine::{tensor_parallel, Executor};
use crate::loader;

/// The scheduler settings every load applies, owned so the load runs on
/// another thread.
pub(crate) struct LoadJob<R: Runtime<DType = DType>> {
    /// The directory models are searched in.
    pub(crate) model_dir: PathBuf,
    /// The device the weights upload to.
    pub(crate) device: R::Device,
    /// The KV cache context size.
    pub(crate) num_ctx: usize,
    /// `--chat-template`: a built-in name or a template file.
    pub(crate) chat_template: Option<String>,
    /// `--mmproj` and the image token limits.
    pub(crate) vision: VisionOverrides,
    /// The user config file's `inference` keys.
    pub(crate) inference_overlay: InferenceOverlay,
    /// The configured tensor parallel degree. `None` takes the model's.
    pub(crate) tensor_parallel_size: Option<usize>,
    /// The configured compute dtype. `None` takes the model's.
    pub(crate) dtype: Option<DtypeRequest>,
}

impl<R: Runtime<DType = DType>> LoadJob<R>
where
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + NormalizationOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + boostr::SamplingOps<R>
        + boostr::GrammarDfaOps<R>
        + ModelClient<R>
        + boostr::quant::DequantOps<R>
        + boostr::quant::QuantMatmulOps<R>,
{
    /// Load the model at `model_path` and start its executor.
    ///
    /// # Errors
    ///
    /// A tensor parallel degree that does not validate or build, or a
    /// model, tokenizer, chat template or vision tower that does not load.
    pub(crate) fn run(self, model_path: &Path) -> Result<Executor<R>> {
        // Tensor parallel degree: BLAZR_TP_SIZE, then the user config file,
        // then the model's own config.json, then 1.
        let tp_size = self.tensor_parallel_size.unwrap_or_else(|| {
            loader::model_config_inference(model_path)
                .map_or(1, |inference| inference.tensor_parallel_size)
        });
        tensor_parallel::validate_tp_config(tp_size, None).map_err(|e| anyhow!("{e}"))?;
        if tp_size > 1 {
            return self.run_tensor_parallel(model_path, tp_size);
        }

        let (model, mut config, tokenizer) =
            load_model_parts::<R>(model_path, &self.device, self.dtype.as_ref(), None)?;
        self.apply_settings(&mut config, tp_size)?;
        let executor = start_executor(
            model_path,
            (model, config, tokenizer),
            self.device,
            self.num_ctx,
            self.chat_template.as_deref(),
        )?;
        Ok(executor.with_model_dir(self.model_dir))
    }

    /// Load `model_path` sharded `tp_size` ways. Rank r runs on GPU
    /// `device + r`, each as a thread of this process. Followers load their
    /// shards while rank 0 does.
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    fn run_tensor_parallel(self, model_path: &Path, tp_size: usize) -> Result<Executor<R>> {
        use crate::model::chat_template::ChatTemplate;
        use crate::tokenizer::checkpoint_tokenizer;
        use boostr::runtime::Device;
        let (tp_state, comm) =
            tensor_parallel::init_tp(tp_size, self.device.id(), model_path, self.dtype)
                .map_err(|e| anyhow!("{e}"))?;
        let (model, mut config) =
            loader::load_model_tp::<R, _>(model_path, &self.device, comm, self.dtype.as_ref())?;
        if let Some(group) = &tp_state.group {
            group.wait_loaded()?;
        }
        self.apply_settings(&mut config, tp_size)?;
        tensor_parallel::restrict_to_lockstep(&mut config.inference);
        let tokenizer = checkpoint_tokenizer(model_path, config.vocab_size())?;
        let chat_template = ChatTemplate::detect(
            model_path,
            config.model_type(),
            self.chat_template.as_deref(),
        )?;
        Ok(Executor::new_tensor_parallel(
            model,
            config,
            tokenizer,
            self.device,
            self.num_ctx,
            tp_state,
        )?
        .with_chat_template(chat_template)
        .with_model_dir(self.model_dir))
    }

    #[cfg(not(all(feature = "cuda", feature = "nccl")))]
    fn run_tensor_parallel(self, _model_path: &Path, tp_size: usize) -> Result<Executor<R>> {
        Err(anyhow!(
            "tensor_parallel_size {tp_size} needs the cuda and nccl features. Rebuild with \
             --features cuda,nccl, or set the size to 1"
        ))
    }

    /// Write the user config file's inference keys, then the command-line
    /// vision flags, over a freshly loaded model's config. Records the
    /// tensor parallel degree the model runs at.
    fn apply_settings(&self, config: &mut BlazrConfig, tp_size: usize) -> Result<()> {
        self.inference_overlay.apply(&mut config.inference)?;
        self.vision.apply(&mut config.inference);
        config.inference.tensor_parallel_size = tp_size;
        Ok(())
    }
}
