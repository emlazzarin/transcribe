#!/usr/bin/env bash
# transcribe installer: prerequisites, API key, build, optional ~/.local/bin copy.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

have()  { command -v "$1" >/dev/null 2>&1; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
ok()    { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn()  { printf '  \033[33m!\033[0m %s\n' "$*"; }
fail()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }

confirm() { # $1 = prompt, $2 = default answer (Y or N)
  local ans
  read -r -p "$1 " ans
  if [ "${2:-Y}" = "Y" ]; then
    [[ "${ans:-Y}" =~ ^[Yy]?$ ]]
  else
    [[ "${ans:-N}" =~ ^[Yy]$ ]]
  fi
}

# Create $1 if missing and tighten to mode 600 — never truncates existing content.
secure_file() { (umask 077; touch "$1"); chmod 600 "$1"; }

bold "transcribe installer"

# ---------------------------------------------------------------- prerequisites
if have go; then
  ok "go $(go env GOVERSION 2>/dev/null | sed 's/^go//') found"
else
  if have brew; then
    confirm "Go is required to build. Install with Homebrew now? [Y/n]" Y || fail "Go is required: https://go.dev/dl/"
    brew install go
  else
    fail "Go is required to build. Install it from https://go.dev/dl/ (or via your package manager) and re-run."
  fi
fi

if have ffmpeg; then
  ok "ffmpeg found"
else
  warn "ffmpeg not found — needed for video (mp4/mov/…) and for WAV/AIFF audio."
  warn "mp3/m4a/flac/ogg files and URLs work without it."
  if have brew; then
    confirm "  Install ffmpeg with Homebrew now? [Y/n]" Y && brew install ffmpeg
  else
    warn "install it later with your package manager (e.g. apt install ffmpeg)"
  fi
fi

# ---------------------------------------------------------------- API key
# keep in sync with firstEnv(...) in main.go
KEY_VARS='(ELEVENLABS_API_KEY|XI_API_KEY|ELEVEN_API_KEY|ELEVENLABS_KEY)'
CONFIG_DIR="$HOME/.config/transcribe"
CONFIG_ENV="$CONFIG_DIR/env"
key_in_file() { [ -f "$1" ] && grep -qE "^(export )?${KEY_VARS}=.." "$1"; }

verify_key() { # $1=key — fed to curl via stdin config so it never appears in argv/ps
  local code
  code=$(printf 'header = "xi-api-key: %s"\n' "$1" | curl -sS -K - -o /dev/null \
    -w '%{http_code}' --max-time 15 https://api.elevenlabs.io/v1/user 2>/dev/null || echo "000")
  [ "$code" = "200" ]
}

if [ -n "${ELEVENLABS_API_KEY:-}" ]; then
  ok "using ELEVENLABS_API_KEY from environment"
elif key_in_file .env; then
  ok "found API key in ./.env"
elif key_in_file "$CONFIG_ENV"; then
  ok "found API key in ~/.config/transcribe/env"
else
  echo
  echo "An ElevenLabs API key is required (create one at https://elevenlabs.io/app/settings/api-keys)."
  read -r -s -p "Paste your ElevenLabs API key (input hidden): " API_KEY
  echo
  [ -n "$API_KEY" ] || fail "no key entered"
  printf 'checking key… '
  if verify_key "$API_KEY"; then
    printf 'valid\n'
  else
    printf 'could not verify\n'
    confirm "  The key didn't verify against the ElevenLabs API. Save it anyway? [y/N]" N || fail "aborted — double-check the key and re-run"
  fi
  secure_file .env
  printf 'ELEVENLABS_API_KEY=%s\n' "$API_KEY" >> .env
  ok "saved to ./.env (mode 600, gitignored)"
fi

# ---------------------------------------------------------------- build
VERSION="$(git describe --tags --always --dirty 2>/dev/null || echo dev)"
echo
printf 'building transcribe %s… ' "$VERSION"
go build -trimpath -ldflags "-s -w -X main.version=$VERSION" -o transcribe .
printf 'done\n'
ok "./transcribe is ready to use in this repo"

# ---------------------------------------------------------------- optional PATH install
echo
BIN="./transcribe"
if confirm "Also copy to ~/.local/bin for use outside this repo? [Y/n]" Y; then
  mkdir -p "$HOME/.local/bin"
  cp transcribe "$HOME/.local/bin/transcribe"
  ok "installed to ~/.local/bin/transcribe"
  # Copy the key to the per-user config for runs outside this checkout.
  if ! key_in_file "$CONFIG_ENV" && key_in_file .env; then
    (umask 077; mkdir -p "$CONFIG_DIR")
    chmod 700 "$CONFIG_DIR"
    secure_file "$CONFIG_ENV"
    grep -E "^(export )?${KEY_VARS}=" .env >> "$CONFIG_ENV"
    ok "API key copied to ~/.config/transcribe/env (mode 600)"
  fi
  case ":$PATH:" in
    *":$HOME/.local/bin:"*) BIN="transcribe" ;; # only recommend the bare name if it will resolve
    *) warn "~/.local/bin is not on your PATH — add this to your shell profile:"
       echo '      export PATH="$HOME/.local/bin:$PATH"' ;;
  esac
fi

if ! have claude && [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${GEMINI_API_KEY:-}" ] && [ -z "${GOOGLE_API_KEY:-}" ]; then
  echo
  warn "optional: speaker naming uses the claude CLI or an ANTHROPIC_API_KEY/GEMINI_API_KEY."
  warn "without one, transcripts still work but use generic 'Speaker N' labels."
fi

echo
bold "all set — try it:"
echo "  $BIN -c \"context about the recording\" your-file.mp4"
