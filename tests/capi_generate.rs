//! The C ABI, called through its `extern "C"` functions on the CPU.
//!
//! The model test loads a Llama-family checkpoint, generates 16 greedy
//! tokens through the token callback, stops one generation from the
//! callback and one through `blazr_cancel`, tokenizes, embeds, and unloads
//! twice. The second unload must report `INVALID_HANDLE`.
//!
//! # Fixture
//!
//! `BLAZR_TEST_LLAMA_MODEL` is a Llama-family SafeTensors directory or GGUF
//! file. Unset, `$BOOSTR_MODELS_DIR/llama-3.2-1b` is tried. The model test
//! prints one skip line and passes when neither is present.
//!
//! ```bash
//! BOOSTR_MODELS_DIR=/path/to/models cargo test --features capi --test capi_generate
//! ```

#![cfg(feature = "capi")]

use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;

use blazr::capi::*;

mod common;
use common::{model_fixture, skip_notice};

const ENV_MODEL: &str = "BLAZR_TEST_LLAMA_MODEL";
const MODELS_DIR_ENTRY: &str = "llama-3.2-1b";
const PROMPT: &str = "The capital of France is";

fn last_error() -> String {
    // SAFETY: blazr_last_error returns a valid NUL-terminated string.
    unsafe { CStr::from_ptr(blazr_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// What the token callback records.
struct Collected {
    pieces: Vec<String>,
    /// Return nonzero at this many tokens.
    stop_at: Option<usize>,
    /// Call `blazr_cancel` on this model at `cancel_at` tokens.
    cancel_model: *mut BlazrModel,
    cancel_at: Option<usize>,
}

impl Collected {
    fn new() -> Self {
        Self {
            pieces: Vec::new(),
            stop_at: None,
            cancel_model: ptr::null_mut(),
            cancel_at: None,
        }
    }
}

unsafe extern "C" fn on_token(piece: *const c_char, len: usize, user_data: *mut c_void) -> i32 {
    // SAFETY: the test passes a `Collected` as user data.
    let collected = unsafe { &mut *user_data.cast::<Collected>() };
    // SAFETY: blazr passes `len` valid bytes plus a NUL.
    let bytes = unsafe { std::slice::from_raw_parts(piece.cast::<u8>(), len) };
    // SAFETY: as above, the NUL follows the bytes.
    assert_eq!(unsafe { *piece.add(len) }, 0, "piece is not NUL-terminated");
    collected
        .pieces
        .push(String::from_utf8(bytes.to_vec()).expect("piece is UTF-8"));
    let count = collected.pieces.len();
    if collected.cancel_at == Some(count) {
        assert_eq!(blazr_cancel(collected.cancel_model), BlazrStatus::Ok);
    }
    i32::from(collected.stop_at == Some(count))
}

fn greedy(max_tokens: u32) -> BlazrGenerateParams {
    BlazrGenerateParams {
        max_tokens,
        temperature: 0.0,
        repeat_penalty: 1.0,
        ..blazr_generate_params_default()
    }
}

/// Generate with `collected` as the callback state. Returns the status and
/// the text from `out_text`.
fn generate(
    model: *mut BlazrModel,
    params: &BlazrGenerateParams,
    collected: &mut Collected,
) -> (BlazrStatus, String) {
    let prompt = CString::new(PROMPT).unwrap();
    let mut text: *mut c_char = ptr::null_mut();
    // SAFETY: every pointer is valid for the call.
    let status = unsafe {
        blazr_generate(
            model,
            prompt.as_ptr(),
            params,
            Some(on_token),
            ptr::from_mut(collected).cast(),
            &mut text,
        )
    };
    let out = if text.is_null() {
        String::new()
    } else {
        // SAFETY: blazr returned a NUL-terminated string.
        let owned = unsafe { CStr::from_ptr(text) }.to_str().unwrap().to_owned();
        // SAFETY: freed once, with the matching free function.
        unsafe { blazr_string_free(text) };
        owned
    };
    (status, out)
}

#[test]
fn version_is_the_crate_version() {
    // SAFETY: a static NUL-terminated string.
    let version = unsafe { CStr::from_ptr(blazr_version()) };
    assert_eq!(version.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
}

#[test]
fn bad_arguments_return_codes_not_panics() {
    let mut model: *mut BlazrModel = ptr::null_mut();
    // SAFETY: a null path is the case under test.
    let status = unsafe { blazr_model_load(ptr::null(), ptr::null(), &mut model) };
    assert_eq!(status, BlazrStatus::InvalidArgument);
    assert!(model.is_null());
    assert!(last_error().contains("path is null"), "{}", last_error());

    let missing = CString::new("/nonexistent/blazr-capi-model").unwrap();
    // SAFETY: valid pointers.
    let status = unsafe { blazr_model_load(missing.as_ptr(), ptr::null(), &mut model) };
    assert_eq!(status, BlazrStatus::Load);
    assert!(last_error().contains("does not exist"), "{}", last_error());

    let device = CString::new("tpu").unwrap();
    let options = BlazrLoadOptions {
        device: device.as_ptr(),
        ..blazr_load_options_default()
    };
    // SAFETY: valid pointers.
    let status = unsafe { blazr_model_load(missing.as_ptr(), &options, &mut model) };
    assert_eq!(status, BlazrStatus::InvalidArgument);
    assert!(
        last_error().contains("'tpu' is not valid"),
        "{}",
        last_error()
    );

    let stale = ptr::without_provenance_mut::<BlazrModel>(usize::MAX);
    assert_eq!(blazr_model_unload(stale), BlazrStatus::InvalidHandle);
    assert_eq!(blazr_cancel(stale), BlazrStatus::InvalidHandle);
    assert!(
        last_error().contains("unknown or already unloaded"),
        "{}",
        last_error()
    );

    // SAFETY: null is a documented no-op for every free function.
    unsafe {
        blazr_string_free(ptr::null_mut());
        blazr_tokens_free(ptr::null_mut(), 0);
        blazr_floats_free(ptr::null_mut(), 0);
    }
}

#[test]
fn load_generate_cancel_unload_on_cpu() {
    let Some(path) = model_fixture(ENV_MODEL, MODELS_DIR_ENTRY) else {
        skip_notice("a Llama-family checkpoint", ENV_MODEL);
        return;
    };
    let path = CString::new(path.to_str().unwrap()).unwrap();
    let device = CString::new("cpu").unwrap();
    let options = BlazrLoadOptions {
        device: device.as_ptr(),
        num_ctx: 256,
        ..blazr_load_options_default()
    };
    let mut model: *mut BlazrModel = ptr::null_mut();
    // SAFETY: valid pointers.
    let status = unsafe { blazr_model_load(path.as_ptr(), &options, &mut model) };
    assert_eq!(status, BlazrStatus::Ok, "load: {}", last_error());
    assert!(!model.is_null());

    // 16 greedy tokens, each through the callback.
    let mut full = Collected::new();
    let (status, text) = generate(model, &greedy(16), &mut full);
    assert_eq!(status, BlazrStatus::Ok, "generate: {}", last_error());
    assert_eq!(full.pieces.len(), 16, "pieces: {:?}", full.pieces);
    assert_eq!(text, full.pieces.concat());
    assert!(!text.trim().is_empty());
    println!("generated: {PROMPT}{text}");

    // The callback stops generation at the third token.
    let mut stopped = Collected::new();
    stopped.stop_at = Some(3);
    let (status, text) = generate(model, &greedy(16), &mut stopped);
    assert_eq!(status, BlazrStatus::Cancelled);
    assert_eq!(stopped.pieces.len(), 3);
    assert_eq!(text, full.pieces[..3].concat(), "greedy decoding repeats");

    // blazr_cancel stops generation after the third token.
    let mut cancelled = Collected::new();
    cancelled.cancel_model = model;
    cancelled.cancel_at = Some(3);
    let (status, _) = generate(model, &greedy(16), &mut cancelled);
    assert_eq!(status, BlazrStatus::Cancelled);
    assert_eq!(cancelled.pieces.len(), 3);

    // A later generation clears the cancel request.
    let mut again = Collected::new();
    let (status, _) = generate(model, &greedy(4), &mut again);
    assert_eq!(status, BlazrStatus::Ok);
    assert_eq!(again.pieces.len(), 4);

    // Tokenize: ids come back in a blazr-owned array.
    let text = CString::new(PROMPT).unwrap();
    let mut ids: *mut u32 = ptr::null_mut();
    let mut len = 0usize;
    // SAFETY: valid pointers.
    let status = unsafe { blazr_tokenize(model, text.as_ptr(), &mut ids, &mut len) };
    assert_eq!(status, BlazrStatus::Ok, "tokenize: {}", last_error());
    assert!(len >= 5, "{len} ids for {PROMPT:?}");
    // SAFETY: the pair blazr_tokenize returned.
    unsafe { blazr_tokens_free(ids, len) };

    // Embed: one unit-length vector.
    let mut values: *mut f32 = ptr::null_mut();
    // SAFETY: valid pointers.
    let status = unsafe { blazr_embed(model, text.as_ptr(), &mut values, &mut len) };
    assert_eq!(status, BlazrStatus::Ok, "embed: {}", last_error());
    // SAFETY: blazr returned `len` floats.
    let vector = unsafe { std::slice::from_raw_parts(values, len) };
    let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "norm {norm}");
    // SAFETY: the pair blazr_embed returned.
    unsafe { blazr_floats_free(values, len) };

    // Unload, then every use of the stale handle fails cleanly.
    assert_eq!(blazr_model_unload(model), BlazrStatus::Ok);
    assert_eq!(blazr_model_unload(model), BlazrStatus::InvalidHandle);
    let (status, _) = generate(model, &greedy(4), &mut Collected::new());
    assert_eq!(status, BlazrStatus::InvalidHandle);
    assert!(
        last_error().contains("already unloaded"),
        "{}",
        last_error()
    );
}
