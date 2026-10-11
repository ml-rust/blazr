# blazr C ABI

The C ABI embeds the blazr engine in an app. The header is [`include/blazr.h`](../../include/blazr.h). `smoke.c` in this directory calls every step: load, generate with a token callback, stop, cancel, and unload.

## API

| Function | Does |
| --- | --- |
| `blazr_version` | Returns the library version. Do not free it. |
| `blazr_load_options_default` | Returns the default load options. |
| `blazr_model_load` | Loads a GGUF file or a SafeTensors directory. Writes a handle. |
| `blazr_model_unload` | Unloads a model. A second unload returns `BLAZR_STATUS_INVALID_HANDLE`. |
| `blazr_generate_params_default` | Returns the CLI's default sampling settings. |
| `blazr_generate` | Generates text. Calls the callback once per token. |
| `blazr_cancel` | Stops the running generation from any thread. |
| `blazr_tokenize` | Encodes text to token ids. |
| `blazr_embed` | Returns one unit-length vector: the mean of the last hidden states. |
| `blazr_last_error` | Returns the message of the last failed call on this thread. |
| `blazr_string_free`, `blazr_tokens_free`, `blazr_floats_free` | Free what blazr returned. |

Rules:

- Every call returns a `BlazrStatus`. Zero and above mean success. `BLAZR_STATUS_CANCELLED` means the callback or `blazr_cancel` stopped generation. The text so far is valid.
- A failed call stores a UTF-8 message. Read it with `blazr_last_error` on the same thread.
- No panic crosses the boundary. A panic returns `BLAZR_STATUS_PANIC`.
- All strings are UTF-8. Free each returned string or array with its matching free function, never with `free()`.
- A handle is an id, never a pointer to memory. A stale handle returns `BLAZR_STATUS_INVALID_HANDLE`.
- The callback piece is valid only during the callback. A piece can be empty while a multi-byte character is incomplete. Return nonzero to stop.
- `device` takes `cpu`, `cuda`, `cuda:N`, `wgpu` or `wgpu:N`. A device needs its Cargo feature in the library build. Null picks GPU 0 of the build, else the CPU.

## Build the library

Build the engine alone. It needs neither `server` nor `cli`:

```bash
cargo build --release --lib --no-default-features --features capi,f16
```

The build writes `target/release/libblazr.a` (static) and `target/release/libblazr.so` (shared). Add `wgpu` to the features for Vulkan or Metal. Add `cuda` for NVIDIA.

## Linux

Tested on x86-64 Linux. `tests/capi_smoke_c.rs` runs these steps.

```bash
cc examples/capi/smoke.c -I include target/release/libblazr.a \
  -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc -o smoke
./smoke /path/to/llama-3.2-1b
```

## Android

Checked with `cargo check` only, on NDK 27.0.12077973 with API level 24. No model has run on a device.

1. Install the Rust target:

   ```bash
   rustup target add aarch64-linux-android
   ```

2. Set the NDK tools and build the shared library:

   ```bash
   NDK=$HOME/Android/Sdk/ndk/27.0.12077973
   TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
   export ANDROID_NDK_HOME=$NDK
   export CC_aarch64_linux_android=$TC/aarch64-linux-android24-clang
   export AR_aarch64_linux_android=$TC/llvm-ar
   export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$TC/aarch64-linux-android24-clang
   cargo build --release --lib --no-default-features --features capi,f16 \
     --target aarch64-linux-android
   ```

3. Copy `target/aarch64-linux-android/release/libblazr.so` to `app/src/main/jniLibs/arm64-v8a/`.
4. Copy `include/blazr.h` next to your JNI source. Write the JNI functions that call it. blazr ships no Kotlin bindings.
5. Load the library in Kotlin with `System.loadLibrary("blazr")` before the first JNI call.
6. Put the model in app storage and pass its absolute path to `blazr_model_load`.

## iOS

Untested. The check host runs Linux and has no Apple SDK. These steps follow the standard Rust `staticlib` flow.

1. Install the Rust targets on macOS:

   ```bash
   rustup target add aarch64-apple-ios aarch64-apple-ios-sim
   ```

2. Build the static library for the device and the simulator:

   ```bash
   cargo build --release --lib --no-default-features --features capi,f16 --target aarch64-apple-ios
   cargo build --release --lib --no-default-features --features capi,f16 --target aarch64-apple-ios-sim
   ```

3. Wrap both in one XCFramework:

   ```bash
   xcodebuild -create-xcframework \
     -library target/aarch64-apple-ios/release/libblazr.a -headers include \
     -library target/aarch64-apple-ios-sim/release/libblazr.a -headers include \
     -output Blazr.xcframework
   ```

4. Add `Blazr.xcframework` to the Xcode target. Add `blazr.h` to the bridging header. blazr ships no Swift bindings.
5. Add `wgpu` to the features for Metal. Link `Metal.framework` and `QuartzCore.framework` then.
