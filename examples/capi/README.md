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

Verified on an Android x86_64 emulator only. No aarch64 hardware or physical device has run blazr.

| Target           | Runtime                    | Result                                  |
| ---------------- | -------------------------- | --------------------------------------- |
| x86_64 emulator  | CPU                        | Pass. Peak RSS 1,952,672 kB.            |
| x86_64 emulator  | wgpu through the host GPU  | Pass. Peak RSS 378,732 kB.              |
| aarch64 hardware | Any                        | Not run.                                |

The model is Llama 3.2 1B Q4_K_M, 807,689,920 bytes, with its embedded tokenizer. The emulator is an Android 36 `google_apis_playstore` x86_64 image on KVM with NDK r27. The CPU output matches the host CPU run of the same program byte for byte. The host CPU run peaks at 1,941,956 kB.

### Build the library for x86_64

Add the Rust target once, set the NDK tools, and build the static library:

```bash
rustup target add x86_64-linux-android
NDK=$HOME/Android/Sdk/ndk/27.0.12077973
TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
export ANDROID_NDK_HOME=$NDK
export CC_x86_64_linux_android=$TC/x86_64-linux-android26-clang
export AR_x86_64_linux_android=$TC/llvm-ar
export CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER=$TC/x86_64-linux-android26-clang
cargo rustc --release --lib --crate-type staticlib --no-default-features --features capi \
  --target x86_64-linux-android -- --print native-static-libs
```

The build writes `target/x86_64-linux-android/release/libblazr.a`. The command prints the system libraries to link: `-ldl -llog -lunwind -lm -lc`.

### Link with 16 KB pages

The Android 36 image uses 16 KB pages. NDK r27 links at 4 KB by default. Without the flag, the loader fails with `CANNOT LINK EXECUTABLE ... empty/missing DT_HASH/DT_GNU_HASH`. Add `-Wl,-z,max-page-size=16384` to every native library for apps that target 16 KB devices.

Link the smoke program:

```bash
$TC/x86_64-linux-android26-clang -O2 -I include examples/capi/smoke.c -o smoke \
  target/x86_64-linux-android/release/libblazr.a \
  -ldl -llog -lunwind -lm -lc -Wl,-z,max-page-size=16384
```

### Build with wgpu

Build with `--features capi,wgpu`, then add `-landroid` to the link after `libblazr.a`:

```bash
cargo rustc --release --lib --crate-type staticlib --no-default-features --features capi,wgpu \
  --target x86_64-linux-android -- --print native-static-libs
$TC/x86_64-linux-android26-clang -O2 -I include examples/capi/smoke.c -o smoke \
  target/x86_64-linux-android/release/libblazr.a -landroid \
  -ldl -llog -lunwind -lm -lc -Wl,-z,max-page-size=16384
```

Start the emulator with `-gpu host` to use the host GPU through Vulkan. With `-gpu swiftshader_indirect`, the adapter opens, but the model load did not finish in about 55 minutes.

### Run on the emulator

Push the executable and the model, then run the program:

```bash
adb push smoke /data/local/tmp/
adb push <model>.gguf /data/local/tmp/
adb shell 'cd /data/local/tmp && ./smoke <model>.gguf'
```

At temperature 0, Llama 3.2 1B Q4_K_M prints this text in 16 tokens:

```
The capital of France is Paris. It is the most visited city in the world. It is the most
```

The program also cancels generation after 3 tokens. A second unload returns an error.

The aarch64 NEON kernels ran only under qemu-user. No aarch64 Android run has happened.

### arm64 app integration (compile check only)

`cargo check` passes on NDK 27.0.12077973 with API level 24. No model has run on an arm64 device or emulator.

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
