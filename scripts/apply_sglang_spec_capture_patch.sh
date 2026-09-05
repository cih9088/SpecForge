#!/usr/bin/env bash
# Apply an ordered stack of SpecForge patches to INSTALLED sglang
# (site-packages).
#
# Patch files are authored against the sglang source tree (git-style paths
# a/python/sglang/srt/...); an installed package drops the python/ prefix, so
# strip TWO components (a/ + python/) and apply from the site-packages parent.
#
# The patch basename is its stable identity. spec-capture.patch remains the
# base patch and keeps the legacy .spec_capture_patch.applied receipt; other
# patches get their own receipt. .spec_capture_patch.order records application
# order, and --reverse is deliberately LIFO. Re-running an unchanged patch is
# idempotent and does not change its position in the stack.
#
# Usage: scripts/apply_sglang_spec_capture_patch.sh
#          [--target v0.5.18|kimi-k3-ee560a2|kimi-k3-9acd9cb|kimi-k3-f8493a4]
#          [--reverse]
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="v0.5.18"
PATCH_TARGET=""
REVERSE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --target requires a value" >&2
                exit 2
            fi
            TARGET="$2"
            shift 2
            ;;
        --reverse)
            REVERSE=1
            shift
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

case "$TARGET" in
    v0.5.18)
        EXPECTED_VERSION_PREFIX="0.5.18"
        PATCH_TARGET="$TARGET"
        ;;
    kimi-k3-ee560a2|kimi-k3-9acd9cb|kimi-k3-f8493a4)
        # Kimi K3's SGLang fork currently reports a base-package version that
        # does not uniquely identify this source revision, so patch --check is
        # the authoritative compatibility gate below.
        EXPECTED_VERSION_PREFIX=""
        # One patch is generated against ee560a2 and compatibility-checked
        # against the original f8493a4 integration point and the 9acd9cb tip.
        # Keep the historical directory name so existing automation remains
        # source-compatible.
        PATCH_TARGET="kimi-k3-f8493a4"
        ;;
    *)
        echo "ERROR: unsupported SGLang patch target: $TARGET" >&2
        exit 2
        ;;
esac

PATCH="${SPECFORGE_SPEC_CAPTURE_PATCH:-$HERE/patches/sglang/$PATCH_TARGET/spec-capture.patch}"
PATCH_NAME="$(basename "$PATCH")"
BASE_PATCH_NAME="spec-capture.patch"

if [[ ! "$PATCH_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: patch basename contains unsupported characters: $PATCH_NAME" >&2
    exit 2
fi

SGL_PARENT="${SPECFORGE_SGLANG_ROOT:-$(python -c 'import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))')}"
SGL_VERSION="${SPECFORGE_SGLANG_VERSION:-$(python -c 'import sglang; print(sglang.__version__)')}"
STATE_ROOT="$SGL_PARENT/sglang"
BASE_RECEIPT="$STATE_ROOT/.spec_capture_patch.applied"
ORDER_FILE="$STATE_ROOT/.spec_capture_patch.order"
SINK="$STATE_ROOT/srt/spec_capture_sink.py"

receipt_for_name() {
    if [[ "$1" == "$BASE_PATCH_NAME" ]]; then
        printf '%s\n' "$BASE_RECEIPT"
    else
        printf '%s\n' "$STATE_ROOT/.spec_capture_patch.$1.applied"
    fi
}

APPLIED_COPY="$(receipt_for_name "$PATCH_NAME")"

if ! command -v git > /dev/null; then
    echo "ERROR: git is required to verify and apply SpecForge patches exactly" >&2
    exit 1
fi

# git apply anchors paths at the discovered worktree root, even with -C. When
# site-packages is nested inside another repository, prepend its worktree-
# relative prefix explicitly so Git does not silently skip the patch paths.
GIT_DIRECTORY=""
if ! GIT_DIRECTORY="$(git -C "$SGL_PARENT" rev-parse --show-prefix 2> /dev/null)"; then
    GIT_DIRECTORY=""
fi

if [[ "$REVERSE" == 0 && ! -f "$PATCH" ]]; then
    echo "ERROR: patch file does not exist: $PATCH" >&2
    exit 1
fi

git_apply() {
    local -a command=(git -C "$SGL_PARENT" apply)
    if [[ -n "$GIT_DIRECTORY" ]]; then
        command+=(--directory="$GIT_DIRECTORY")
    fi
    "${command[@]}" "$@"
}

check_apply() {
    git_apply --check -p2 "$1" 2> /dev/null
}

check_reverse() {
    git_apply --reverse --check -p2 "$1" 2> /dev/null
}

apply_exact() {
    git_apply -p2 "$1"
}

reverse_exact() {
    git_apply --reverse -p2 "$1"
}

fail_unknown_state() {
    echo "ERROR: $STATE_ROOT carries an unknown spec-capture patch state for $PATCH_NAME" >&2
    echo "reinstall sglang (or clear the cached venv) and re-run patches in application order" >&2
    exit 1
}

fail_order() {
    echo "ERROR: spec-capture patch order violation for $PATCH_NAME: $1" >&2
    exit 1
}

declare -a PATCH_ORDER=()

order_index() {
    local wanted="$1"
    local index
    for index in "${!PATCH_ORDER[@]}"; do
        if [[ "${PATCH_ORDER[$index]}" == "$wanted" ]]; then
            printf '%s\n' "$index"
            return 0
        fi
    done
    return 1
}

write_order() {
    if [[ ${#PATCH_ORDER[@]} -eq 0 ]]; then
        rm -f "$ORDER_FILE"
        return
    fi

    local temporary="$ORDER_FILE.tmp.$$"
    printf '%s\n' "${PATCH_ORDER[@]}" > "$temporary"
    mv "$temporary" "$ORDER_FILE"
}

write_receipt() {
    local source="$1"
    local temporary="$APPLIED_COPY.tmp.$$"
    cp "$source" "$temporary"
    mv "$temporary" "$APPLIED_COPY"
}

load_order() {
    local entry
    local existing
    local receipt

    if [[ -f "$ORDER_FILE" ]]; then
        while IFS= read -r entry || [[ -n "$entry" ]]; do
            if [[ ! "$entry" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
                fail_unknown_state
            fi
            for existing in "${PATCH_ORDER[@]}"; do
                if [[ "$existing" == "$entry" ]]; then
                    fail_unknown_state
                fi
            done
            receipt="$(receipt_for_name "$entry")"
            if [[ ! -f "$receipt" ]]; then
                fail_unknown_state
            fi
            PATCH_ORDER+=("$entry")
        done < "$ORDER_FILE"

        if [[ ${#PATCH_ORDER[@]} -eq 0 || "${PATCH_ORDER[0]}" != "$BASE_PATCH_NAME" ]]; then
            fail_unknown_state
        fi
    elif [[ -f "$BASE_RECEIPT" ]]; then
        # Existing installations used one receipt for the base patch. Promote
        # it to the bottom of the ordered stack without changing its contents.
        PATCH_ORDER=("$BASE_PATCH_NAME")
        write_order
        echo "migrated legacy spec-capture patch receipt to ordered state"
    fi
}

ensure_no_unrelated_orphan_receipts() {
    local receipt
    local name
    local found
    local entry
    local -a receipts=("$STATE_ROOT"/.spec_capture_patch.*.applied)

    for receipt in "${receipts[@]}"; do
        [[ -e "$receipt" ]] || continue
        name="${receipt#"$STATE_ROOT/.spec_capture_patch."}"
        name="${name%.applied}"
        [[ "$name" == "$PATCH_NAME" ]] && continue
        found=0
        for entry in "${PATCH_ORDER[@]}"; do
            if [[ "$entry" == "$name" ]]; then
                found=1
                break
            fi
        done
        if [[ "$found" == 0 ]]; then
            fail_unknown_state
        fi
    done
}

ensure_stack_active_through() {
    local last_index="$1"
    local index
    local receipt
    for ((index = 0; index <= last_index; index++)); do
        receipt="$(receipt_for_name "${PATCH_ORDER[$index]}")"
        if ! check_reverse "$receipt"; then
            fail_order "apply or recover ${PATCH_ORDER[$index]} first"
        fi
    done
}

ensure_recovery_order() {
    local selected_index="$1"
    local index
    local receipt

    if ((selected_index > 0)); then
        ensure_stack_active_through "$((selected_index - 1))"
    fi

    for ((index = selected_index + 1; index < ${#PATCH_ORDER[@]}; index++)); do
        receipt="$(receipt_for_name "${PATCH_ORDER[$index]}")"
        if check_reverse "$receipt"; then
            fail_order "reverse ${PATCH_ORDER[$index]} before recovering $PATCH_NAME"
        fi
        if ! check_apply "$receipt"; then
            fail_unknown_state
        fi
    done
}

# Pip can replace package-owned files while leaving patch receipts and files
# added by the base patch. Only base-patch recovery may move the known sink;
# add-on recovery must never disturb it.
recover_selected_patch() {
    if [[ -z "$EXPECTED_VERSION_PREFIX" || "$SGL_VERSION" != "$EXPECTED_VERSION_PREFIX"* ]]; then
        return 1
    fi

    local stale_sink=""
    if [[ "$PATCH_NAME" == "$BASE_PATCH_NAME" && -f "$SINK" ]]; then
        stale_sink="$SINK.specforge-stale.$$"
        if [[ -e "$stale_sink" ]]; then
            return 1
        fi
        mv "$SINK" "$stale_sink"
    fi

    if check_apply "$PATCH"; then
        if apply_exact "$PATCH"; then
            [[ -z "$stale_sink" ]] || rm -f "$stale_sink"
            write_receipt "$PATCH"
            echo "recovered stale $PATCH_NAME files left by a cached pip upgrade"
            return 0
        fi
    fi

    if [[ -n "$stale_sink" && -f "$stale_sink" ]]; then
        mv "$stale_sink" "$SINK"
    fi
    return 1
}

prove_recorded_patch_absent() {
    local receipt="$1"
    if check_apply "$receipt"; then
        return 0
    fi

    if [[ "$PATCH_NAME" != "$BASE_PATCH_NAME" || ! -f "$SINK" ]]; then
        return 1
    fi

    local stale_sink="$SINK.specforge-stale.$$"
    if [[ -e "$stale_sink" ]]; then
        return 1
    fi
    mv "$SINK" "$stale_sink"
    if check_apply "$receipt"; then
        rm -f "$stale_sink"
        return 0
    fi
    mv "$stale_sink" "$SINK"
    return 1
}

ensure_can_append_selected_patch() {
    if [[ "$PATCH_NAME" == "$BASE_PATCH_NAME" ]]; then
        if [[ ${#PATCH_ORDER[@]} -ne 0 ]]; then
            fail_order "$BASE_PATCH_NAME must remain the first stack entry"
        fi
    else
        if [[ ${#PATCH_ORDER[@]} -eq 0 || "${PATCH_ORDER[0]}" != "$BASE_PATCH_NAME" ]]; then
            fail_order "apply $BASE_PATCH_NAME first"
        fi
        ensure_stack_active_through "$((${#PATCH_ORDER[@]} - 1))"
    fi
}

append_selected_patch() {
    ensure_can_append_selected_patch
    PATCH_ORDER+=("$PATCH_NAME")
    write_order
}

load_order
ensure_no_unrelated_orphan_receipts

if [[ -n "$EXPECTED_VERSION_PREFIX" && "$SGL_VERSION" != "$EXPECTED_VERSION_PREFIX"* ]]; then
    echo "WARNING: installed sglang is $SGL_VERSION; the patch targets $TARGET" >&2
fi

SELECTED_INDEX=-1
if found_index="$(order_index "$PATCH_NAME")"; then
    SELECTED_INDEX="$found_index"
fi

# Reconcile the only safe orphan-receipt state: the patch was applied and its
# receipt was written, but the process stopped before appending the order file.
if [[ "$SELECTED_INDEX" == -1 && -f "$APPLIED_COPY" ]]; then
    if check_reverse "$APPLIED_COPY"; then
        append_selected_patch
        SELECTED_INDEX="$((${#PATCH_ORDER[@]} - 1))"
        echo "adopted $PATCH_NAME receipt into ordered state"
    elif check_apply "$APPLIED_COPY"; then
        rm -f "$APPLIED_COPY"
    else
        fail_unknown_state
    fi
fi

if [[ "$REVERSE" == 1 ]]; then
    if [[ "$SELECTED_INDEX" == -1 ]]; then
        if [[ ! -f "$PATCH" ]]; then
            echo "ERROR: no receipt or patch file is available for $PATCH_NAME --reverse" >&2
            exit 1
        fi
        ensure_can_append_selected_patch
        if check_reverse "$PATCH"; then
            write_receipt "$PATCH"
            append_selected_patch
            SELECTED_INDEX="$((${#PATCH_ORDER[@]} - 1))"
            echo "adopted already-applied $PATCH_NAME before reverse"
        else
            fail_unknown_state
        fi
    fi

    TOP_INDEX="$((${#PATCH_ORDER[@]} - 1))"
    if [[ "$SELECTED_INDEX" != "$TOP_INDEX" ]]; then
        fail_order "reverse ${PATCH_ORDER[$TOP_INDEX]} first"
    fi

    if check_reverse "$APPLIED_COPY"; then
        reverse_exact "$APPLIED_COPY"
    elif prove_recorded_patch_absent "$APPLIED_COPY"; then
        echo "discarded stale $PATCH_NAME receipt after cached package replacement"
    else
        fail_unknown_state
    fi

    unset 'PATCH_ORDER[TOP_INDEX]'
    write_order
    rm -f "$APPLIED_COPY"
    echo "spec-capture patch $PATCH_NAME ($TARGET) --reverse at $STATE_ROOT (sglang $SGL_VERSION)"
    exit 0
fi

if [[ "$SELECTED_INDEX" == -1 ]]; then
    ensure_can_append_selected_patch
    if check_reverse "$PATCH"; then
        write_receipt "$PATCH"
        append_selected_patch
        echo "spec-capture patch $PATCH_NAME ($TARGET) already applied at $STATE_ROOT (adopted)"
        exit 0
    fi
    if ! check_apply "$PATCH"; then
        fail_unknown_state
    fi
    apply_exact "$PATCH"
    write_receipt "$PATCH"
    append_selected_patch
    echo "spec-capture patch $PATCH_NAME ($TARGET) applied at $STATE_ROOT (sglang $SGL_VERSION)"
    exit 0
fi

TOP_INDEX="$((${#PATCH_ORDER[@]} - 1))"
if cmp -s "$APPLIED_COPY" "$PATCH" && check_reverse "$APPLIED_COPY"; then
    echo "spec-capture patch $PATCH_NAME ($TARGET) already applied at $STATE_ROOT"
    exit 0
fi

if ! cmp -s "$APPLIED_COPY" "$PATCH"; then
    if [[ "$SELECTED_INDEX" != "$TOP_INDEX" ]]; then
        fail_order "reverse patches above $PATCH_NAME before updating it"
    fi

    if check_reverse "$APPLIED_COPY"; then
        echo "$PATCH_NAME changed; reversing the recorded version first"
        reverse_exact "$APPLIED_COPY"
        if check_apply "$PATCH" && apply_exact "$PATCH"; then
            write_receipt "$PATCH"
            echo "spec-capture patch $PATCH_NAME ($TARGET) updated at $STATE_ROOT"
            exit 0
        fi

        if check_apply "$APPLIED_COPY" && apply_exact "$APPLIED_COPY"; then
            echo "ERROR: updated $PATCH_NAME does not apply; restored the recorded version" >&2
            exit 1
        fi
        fail_unknown_state
    fi
fi

ensure_recovery_order "$SELECTED_INDEX"
if recover_selected_patch; then
    exit 0
fi
fail_unknown_state
