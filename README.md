# transcribe

Fast, diarized, timestamped, cleaned-up transcripts from almost anything that
contains speech: a media file in any format ffmpeg reads (mp3, mp4, mov, wav,
mkv, voice memos, …), a URL (YouTube, TikTok, or any hosted audio/video), or
stdin, or a `.json` saved by `--raw` (reprocessed without another API call).
Each input yields a `.txt` transcript beside it (in the current directory for
URLs and stdin); `-o` overrides.

```
transcribe episode.mp4
transcribe https://www.youtube.com/watch?v=sWGfAyeBqzg    # URLs fetched server-side
transcribe -c "The Rest Is History; hosts Tom Holland and Dominic Sandbrook" episode.mp3
transcribe --srt *.mp3                                    # parallel, with subtitles
some-pipeline | transcribe -                              # read media from stdin
```

Example output — a real excerpt from the URL above (a 62-minute episode of *The
Rest Is History*, transcribed in under 3 minutes; both hosts identified from the
audio alone):

```
[0:54:24] Dominic Sandbrook: It's interesting. That philosophy is born in
opposition to story, right? That's basically what you're suggesting.

[0:54:35] Tom Holland: It absolutely is. And the most famous philosopher of
all, who was a young man in his 20s when the Bacchae was staged [...] goes so
far as to argue that in an ideal state, the poets, Homer, Hesiod, and so on,
should be banned. And that man, of course, is Plato.
```

## Install

```
git clone https://github.com/emlazzarin/transcribe.git && cd transcribe
./install.sh        # or: bash install.sh
```

The installer checks for Go and ffmpeg (offering to `brew install` them), prompts for
your [ElevenLabs API key](https://elevenlabs.io/app/settings/api-keys) (verified
against the API, saved to a gitignored `.env` with mode 600), builds `./transcribe`,
and optionally installs it to `~/.local/bin` so it works from any directory.

Instead of the installer, Go users can `go install github.com/emlazzarin/transcribe@latest`
(Go 1.24+), then provide the key via the environment or `~/.config/transcribe/env`.

Manual alternative: `go build -o transcribe .` and put `ELEVENLABS_API_KEY=...` in
`.env`, the environment, or `~/.config/transcribe/env`.

Tagged releases build prebuilt binaries for macOS/Linux/Windows (amd64 + arm64) via
GitHub Actions + GoReleaser — see the Releases page if you don't want a Go toolchain.
On Windows, set the key in the environment (or `%USERPROFILE%\.config\transcribe\env`)
and make sure ffmpeg is on PATH for video inputs.

## How it works

1. **Extract** — ffmpeg stream-copies the audio track out of the container (no
   re-encode) and pipes it *directly into the upload*, so extraction and upload
   overlap with no temp files. MP3/M4A/FLAC/OGG inputs upload as-is; uncompressed
   audio is losslessly FLAC-encoded on the fly. URL inputs (YouTube, TikTok, or
   any hosted media) skip all of this — ElevenLabs fetches them server-side.
2. **Transcribe** — [ElevenLabs Scribe v2](https://elevenlabs.io/docs/overview/models#scribe-v2)
   with speaker diarization, word-level timestamps, server-side disfluency removal
   (no "ums", false starts), and keyterm biasing for proper nouns.
3. **Name & correct** (optional) — a small LLM pass reads the transcript and returns
   only a tiny JSON payload: a speaker-id → name map and proper-noun corrections.
   The goal is a best-effort naming and spelling layer on top of a transcript that
   stays what Scribe heard: the payload is applied as exact, word-boundary string
   replacements — no free-form rewrite — and every applied name and correction is
   printed, so you can see exactly what changed. The pass runs on whichever LLM
   backend is available (see [LLM backends](#llm-backends)); with none available
   (or `--no-llm`) you still get the full transcript, speakers just labeled
   `Speaker 1`, `Speaker 2`, …

## Flags

| flag | |
|---|---|
| `-c`, `--context` | Context prompt, fed to both stages: capitalized terms bias recognition (keyterms), and the full text guides speaker naming — naming the expected speakers helps the pass label them. `@file.txt` reads from a file. |
| `-o`, `--output` | Output path (default: input name with `.txt`). An existing directory — or any path with multiple inputs — receives one file per input. |
| `--keyterms` | Comma-separated extra bias terms. |
| `--language` | ISO language hint (default: auto-detect). |
| `--speakers` | Max speaker count hint (1–32). |
| `--srt`, `--vtt` | Also write subtitle files (speaker-labeled, ≤2-line cues). |
| `--verbatim` | Keep filler words / false starts. |
| `--no-llm` | Skip the speaker-naming/correction pass. |
| `--llm-model` | Model for the LLM pass. The value must be valid for the active backend — see [LLM backends](#llm-backends). |
| `--raw <path>` | Also save the raw Scribe JSON to this path (single input only). Pass that `.json` back as an input later to reprocess it — see below. |
| `-q`, `--quiet` | Only print the output path. |
| `--version` | Print version. |

## Naming speakers

Speakers are named from evidence: introductions, how speakers address each
other, the file name, and anything in `--context`. For known speakers, provide
the names and pin the speaker count:

```
transcribe --speakers 2 -c "The Rest Is History; hosts Tom Holland and Dominic Sandbrook" episode.mp3
```

The names also bias recognition toward correct spelling in the transcript body;
`--speakers` keeps one voice from being split into two. The pass is instructed
to leave uncertain speakers generic rather than guess.

## LLM backends

The naming/correction pass uses the first backend available, in this order:

| # | Backend | Enabled by | Default model |
|---|---|---|---|
| 1 | OpenAI API | `OPENAI_API_KEY` | `gpt-5.6-terra` |
| 2 | Anthropic API | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` |
| 3 | Gemini API | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `gemini-2.5-flash` |
| 4 | Codex CLI | `codex` on `PATH` | `gpt-5.6-sol` |
| 5 | Claude CLI | `claude` on `PATH` | `claude-haiku-4-5-20251001` |

`--llm-model` overrides the default, but its value is passed straight to
whichever backend is active — so pass a model that backend understands (an
OpenAI model id when using the OpenAI key, a Gemini model id under the Gemini
key, and so on). It is not a way to pick the backend; that's decided by the
list above. Example: with `OPENAI_API_KEY` set,
`--llm-model gpt-5.6-terra` selects that OpenAI model, whereas
`--llm-model claude-haiku-4-5-20251001` would fail — wrong provider.

The pass is cheap and its output is tiny; any of these models handles it well.
The API backends need only their key in the environment; the CLI backends must
be installed and already logged in.

## Reprocessing a saved run

Transcription is the slow, metered step; the LLM naming pass and subtitle
formatting are neither. Save the raw Scribe response once, then re-run the cheap
half as many times as you like — no second API call, no ElevenLabs key needed:

```
transcribe episode.mp4 --raw episode.json               # transcribe once, keep the raw response
transcribe episode.json -c "hosts Peter and Paul" --srt # re-name speakers + build subtitles, offline
transcribe episode.json --llm-model gpt-5.6-terra       # try a different model on the same audio
```

Any `.json` input is treated as a saved response and reprocessed from disk. What
you can change on reprocessing is everything downstream of Scribe — `--context`,
`--llm-model`, `--no-llm`, `--srt`/`--vtt`, `-o`. Choices baked into the audio
pass itself (`--language`, `--speakers`, `--verbatim`) are fixed in the saved
response; change those and re-transcribe from the media.

## Notes

- **Costs:** ElevenLabs bills transcription per audio-hour on your plan. Keyterm
  biasing adds a 20% surcharge and is only sent when you provide `--context` or
  `--keyterms`. The naming pass makes one (billable) LLM call per file unless it
  uses your local `claude` CLI or you pass `--no-llm`.
- Long monologues are split into paragraphs (~2 min) at sentence boundaries, each
  with a fresh timestamp.
- Multiple inputs are transcribed in parallel (4 at a time), each line of progress
  prefixed with the file name.
- Stdin input (`transcribe -`) writes `./stdin.txt` unless `-o` is given. A pipe
  can't be replayed, so transient upload failures aren't retried for stdin.
- API key lookup order: environment → `./.env` → input file's directory `.env` →
  `~/.config/transcribe/env`.

## License

[MIT](LICENSE)
