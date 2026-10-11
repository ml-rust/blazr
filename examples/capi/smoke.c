/*
 * C smoke test for the blazr C ABI.
 *
 * Loads a model on the CPU, generates 16 greedy tokens through the token
 * callback, stops one generation from the callback and one through
 * blazr_cancel, then unloads twice. Exit code 0 means every step passed.
 *
 * Build: see README.md in this directory.
 * Run:   ./smoke /path/to/llama-3.2-1b
 */
#include <stdio.h>
#include <string.h>

#include "blazr.h"

static const char *PROMPT = "The capital of France is";

typedef struct {
    int count;
    int stop_at;               /* return nonzero at this many tokens; 0 = never */
    int cancel_at;             /* call blazr_cancel at this many tokens; 0 = never */
    BlazrModel *model;
} TokenState;

static int32_t on_token(const char *piece, size_t len, void *user_data) {
    TokenState *state = (TokenState *)user_data;
    state->count += 1;
    fwrite(piece, 1, len, stdout);
    fflush(stdout);
    if (state->cancel_at != 0 && state->count == state->cancel_at) {
        blazr_cancel(state->model);
    }
    return state->stop_at != 0 && state->count >= state->stop_at;
}

static int fail(const char *step, BlazrStatus status) {
    fprintf(stderr, "\n%s failed with status %d: %s\n", step, (int)status, blazr_last_error());
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model path>\n", argv[0]);
        return 2;
    }
    printf("blazr %s\n", blazr_version());

    BlazrLoadOptions options = blazr_load_options_default();
    options.device = "cpu";
    options.num_ctx = 256;
    BlazrModel *model = NULL;
    BlazrStatus status = blazr_model_load(argv[1], &options, &model);
    if (status != BLAZR_STATUS_OK) {
        return fail("blazr_model_load", status);
    }

    BlazrGenerateParams params = blazr_generate_params_default();
    params.max_tokens = 16;
    params.temperature = 0.0f;
    params.repeat_penalty = 1.0f;

    /* 16 tokens, streamed through the callback. */
    TokenState full = {0, 0, 0, model};
    char *text = NULL;
    printf("generated: %s", PROMPT);
    status = blazr_generate(model, PROMPT, &params, on_token, &full, &text);
    printf("\n");
    if (status != BLAZR_STATUS_OK) {
        return fail("blazr_generate", status);
    }
    if (full.count != 16 || text == NULL || strlen(text) == 0) {
        fprintf(stderr, "expected 16 tokens and text, got %d tokens\n", full.count);
        return 1;
    }
    printf("tokens: %d, text bytes: %zu\n", full.count, strlen(text));
    blazr_string_free(text);

    /* The callback stops generation at the third token. */
    TokenState stopped = {0, 3, 0, model};
    printf("stopped: ");
    status = blazr_generate(model, PROMPT, &params, on_token, &stopped, NULL);
    printf("\n");
    if (status != BLAZR_STATUS_CANCELLED || stopped.count != 3) {
        fprintf(stderr, "callback stop: status %d after %d tokens\n", (int)status, stopped.count);
        return 1;
    }
    printf("callback stop after %d tokens\n", stopped.count);

    /* blazr_cancel stops generation after the third token. */
    TokenState cancelled = {0, 0, 3, model};
    printf("cancelled: ");
    status = blazr_generate(model, PROMPT, &params, on_token, &cancelled, NULL);
    printf("\n");
    if (status != BLAZR_STATUS_CANCELLED || cancelled.count != 3) {
        fprintf(stderr, "blazr_cancel: status %d after %d tokens\n", (int)status, cancelled.count);
        return 1;
    }
    printf("cancelled after %d tokens\n", cancelled.count);

    status = blazr_model_unload(model);
    if (status != BLAZR_STATUS_OK) {
        return fail("blazr_model_unload", status);
    }
    status = blazr_model_unload(model);
    if (status != BLAZR_STATUS_INVALID_HANDLE) {
        fprintf(stderr, "second unload returned %d, expected INVALID_HANDLE\n", (int)status);
        return 1;
    }
    printf("second unload: %s\n", blazr_last_error());
    printf("ok\n");
    return 0;
}
