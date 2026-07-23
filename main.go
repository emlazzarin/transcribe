// transcribe: fast diarized, timestamped, cleaned-up transcripts for media
// files, URLs, and stdin via ElevenLabs Scribe v2, with an optional LLM pass
// for speaker naming and proper-noun corrections.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path"
	"path/filepath"
	"regexp"
	"runtime/debug"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unicode/utf8"
)

const (
	sttEndpoint        = "https://api.elevenlabs.io/v1/speech-to-text"
	sttModel           = "scribe_v2"
	maxTurnSeconds     = 120.0 // split a single speaker's turn into paragraphs beyond this
	maxPromptChars     = 160_000
	maxParallel        = 4 // concurrent transcriptions; Scribe allows >= 8 on every plan

	// Default model per LLM backend, used when --llm-model is not given. The
	// --llm-model value is passed verbatim to whichever backend is active, so
	// valid overrides are that provider's model IDs / CLI model names.
	defaultOpenAIModel = "gpt-5.6-terra"
	defaultClaudeModel = "claude-haiku-4-5-20251001" // Anthropic API and claude CLI
	defaultGeminiModel = "gemini-2.5-flash"
	defaultCodexModel  = "gpt-5.6-sol"
)

// version is stamped at build time via -ldflags "-X main.version=...".
var version = "dev"

type config struct {
	output   string
	context  string
	keyterms string
	language string
	speakers int
	verbatim bool
	noLLM    bool
	srt      bool
	vtt      bool
	rawPath  string
	llmModel string
	quiet    bool
}

type logFn func(format string, a ...any)

const usageText = `transcribe — diarized, timestamped, cleaned transcripts from audio, video, or URLs

usage:
  transcribe [flags] <media file, URL, or - for stdin> ...

Inputs may be media files in any format ffmpeg reads, YouTube/TikTok/hosted-media
URLs (fetched server-side by ElevenLabs), - for stdin, or a .json saved by --raw
(reprocessed from disk without calling the API again). Multiple inputs are
transcribed in parallel. Flags may appear before or after inputs; -- ends flags.

flags:
  -c, --context <text>    context to aid transcription & speaker naming; name
                          the people you expect ("hosts Jane Doe and John
                          Smith") to get named speaker labels, plus any terms
                          worth spelling right; @file reads it from a file
  -o, --output <path>     output .txt path (default: alongside input); an
                          existing directory — or any path with multiple
                          inputs — receives one .txt per input
      --keyterms <a,b,…>  comma-separated terms to bias recognition toward
                          (added to those mined from --context)
      --language <code>   ISO language code hint (default: auto-detect)
      --speakers <n>      max number of speakers hint, 1-32 (pairs well with
                          naming the speakers in --context)
      --srt               also write an .srt subtitle file
      --vtt               also write a .vtt subtitle file
      --verbatim          keep filler words and false starts (disable
                          server-side cleanup)
      --no-llm            skip the LLM speaker-naming / proper-noun pass
      --llm-model <name>  model for the LLM pass; value must match the active
                          backend (OpenAI key › Anthropic key › Gemini key ›
                          codex CLI › claude CLI). See "LLM backends" in README
      --raw <path>        also save the raw Scribe JSON to this path (single
                          input only); pass that .json back as an input later to
                          rerun the LLM pass or rebuild subtitles for free
  -q, --quiet             only print the output path
      --version           print version and exit
  -h, --help              show this help

examples:
  transcribe episode.mp4
  transcribe https://www.youtube.com/watch?v=sWGfAyeBqzg
  transcribe -c "The Rest Is History; hosts Tom Holland and Dominic Sandbrook" episode.mp3
  transcribe --srt --vtt lecture.mov
  some-pipeline | transcribe -
  transcribe episode.mp4 --raw episode.json           # keep the raw response
  transcribe episode.json -c "host: Jane Doe" --srt   # reprocess it, no API call
`

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, inputs, err := parseFlags(os.Args[1:])
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	loadDotEnv(".env")
	seenDirs := map[string]bool{}
	for _, in := range inputs {
		if in == "-" || isURL(in) {
			continue
		}
		if dir := filepath.Dir(in); !seenDirs[dir] {
			seenDirs[dir] = true
			loadDotEnv(filepath.Join(dir, ".env"))
		}
	}
	if home, err := os.UserHomeDir(); err == nil {
		loadDotEnv(filepath.Join(home, ".config", "transcribe", "env"))
	}
	// Saved-response inputs never call Scribe, so the API key is only required
	// when at least one input actually needs transcribing.
	needsAPI := false
	for _, in := range inputs {
		if !isSavedResponse(in) {
			needsAPI = true
			break
		}
	}
	apiKey := firstEnv("ELEVENLABS_API_KEY", "XI_API_KEY", "ELEVEN_API_KEY", "ELEVENLABS_KEY")
	if needsAPI && apiKey == "" {
		return errors.New("no ElevenLabs API key found: set ELEVENLABS_API_KEY (or XI_API_KEY) in the environment or a .env file")
	}

	keyterms := collectKeyterms(cfg.keyterms, cfg.context)
	if needsAPI && len(keyterms) > 0 && !cfg.quiet {
		fmt.Fprintf(os.Stderr, "▸ biasing with %d keyterms\n", len(keyterms))
	}

	jobs, err := planJobs(cfg, inputs)
	if err != nil {
		return err
	}

	if len(jobs) == 1 {
		return transcribeFile(ctx, cfg, jobs[0], apiKey, keyterms)
	}

	// Live progress is disabled for parallel runs to keep output readable.
	sem := make(chan struct{}, maxParallel)
	var wg sync.WaitGroup
	var failed atomic.Int64
	for _, jb := range jobs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			if err := transcribeFile(ctx, cfg, jb, apiKey, keyterms); err != nil {
				failed.Add(1)
				fmt.Fprintf(os.Stderr, "[%s] ✗ %v\n", jb.name, err)
			}
		}()
	}
	wg.Wait()
	if n := failed.Load(); n > 0 {
		return fmt.Errorf("%d of %d inputs failed", n, len(jobs))
	}
	return nil
}

type job struct {
	input   string
	name    string
	outPath string
	logf    logFn
	multi   bool
}

// planJobs resolves and validates every destination before any (billable)
// transcription starts: duplicate outputs, --raw races, and unwritable
// directories all fail here instead of after the API call.
func planJobs(cfg *config, inputs []string) ([]job, error) {
	multi := len(inputs) > 1
	if cfg.rawPath != "" && multi {
		return nil, errors.New("--raw supports a single input (parallel runs would overwrite it)")
	}
	jobs := make([]job, len(inputs))
	claimed := map[string]string{}
	for i, in := range inputs {
		p := filepath.Clean(outputPath(cfg.output, in, multi))
		if prev, dup := claimed[p]; dup {
			return nil, fmt.Errorf("%q and %q would both write %s — pass -o with distinct paths", prev, in, p)
		}
		claimed[p] = in
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return nil, fmt.Errorf("creating output directory: %w", err)
		}
		name := displayBase(in)
		prefix := ""
		if multi {
			prefix = "[" + name + "] "
		}
		jobs[i] = job{input: in, name: name, outPath: p, logf: newLogf(cfg.quiet, prefix), multi: multi}
	}
	return jobs, nil
}

func newLogf(quiet bool, prefix string) logFn {
	if quiet {
		return func(string, ...any) {}
	}
	return func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, prefix+format+"\n", a...)
	}
}

func parseFlags(args []string) (*config, []string, error) {
	cfg := &config{}
	fs := flag.NewFlagSet("transcribe", flag.ContinueOnError)
	fs.SetOutput(io.Discard) // errors and usage are printed by us, exactly once
	// All help text lives in usageText; short and long forms share a variable.
	showVersion := fs.Bool("version", false, "")
	fs.StringVar(&cfg.output, "o", "", "")
	fs.StringVar(&cfg.output, "output", "", "")
	fs.StringVar(&cfg.context, "c", "", "")
	fs.StringVar(&cfg.context, "context", "", "")
	fs.StringVar(&cfg.keyterms, "keyterms", "", "")
	fs.StringVar(&cfg.language, "language", "", "")
	fs.IntVar(&cfg.speakers, "speakers", 0, "")
	fs.BoolVar(&cfg.verbatim, "verbatim", false, "")
	fs.BoolVar(&cfg.noLLM, "no-llm", false, "")
	fs.BoolVar(&cfg.srt, "srt", false, "")
	fs.BoolVar(&cfg.vtt, "vtt", false, "")
	fs.StringVar(&cfg.rawPath, "raw", "", "")
	fs.StringVar(&cfg.llmModel, "llm-model", "", "")
	fs.BoolVar(&cfg.quiet, "q", false, "")
	fs.BoolVar(&cfg.quiet, "quiet", false, "")

	// Flags that consume a following argument — everything non-boolean.
	valueFlags := map[string]bool{}
	fs.VisitAll(func(f *flag.Flag) {
		if b, ok := f.Value.(interface{ IsBoolFlag() bool }); !ok || !b.IsBoolFlag() {
			valueFlags[f.Name] = true
		}
	})

	err := fs.Parse(permuteArgs(args, valueFlags))
	switch {
	case errors.Is(err, flag.ErrHelp):
		fmt.Print(usageText)
		os.Exit(0)
	case err != nil:
		return nil, nil, fmt.Errorf("%w (run 'transcribe --help')", err)
	}
	if *showVersion {
		fmt.Printf("transcribe %s\n", buildVersion())
		os.Exit(0)
	}
	if fs.NArg() == 0 { // bare invocation is a help request, not an error
		fmt.Print(usageText)
		os.Exit(0)
	}
	if cfg.speakers != 0 && (cfg.speakers < 1 || cfg.speakers > 32) {
		return nil, nil, fmt.Errorf("--speakers must be between 1 and 32 (got %d)", cfg.speakers)
	}
	if strings.HasPrefix(cfg.context, "@") {
		data, err := os.ReadFile(cfg.context[1:])
		if err != nil {
			return nil, nil, fmt.Errorf("reading context file: %w", err)
		}
		cfg.context = string(data)
	}
	return cfg, fs.Args(), nil
}

// permuteArgs reorders args so flags may appear after positional inputs
// (`transcribe file.mp4 --srt`), matching GNU convention — the stdlib flag
// package stops at the first positional. "--" ends flag parsing; a lone "-"
// is the stdin input. valueFlags lists flags that consume a following argument.
func permuteArgs(args []string, valueFlags map[string]bool) []string {
	var flags, positional []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "--":
			positional = append(positional, args[i+1:]...)
			i = len(args)
		case len(a) > 1 && a[0] == '-':
			flags = append(flags, a)
			name, _, hasValue := strings.Cut(strings.TrimLeft(a, "-"), "=")
			if !hasValue && valueFlags[name] && i+1 < len(args) {
				i++
				flags = append(flags, args[i])
			}
		default:
			positional = append(positional, a)
		}
	}
	return append(append(flags, "--"), positional...)
}

// buildVersion prefers the ldflags-stamped version (installer, releases) and
// falls back to the module version for plain `go install ...@vX.Y.Z` builds.
func buildVersion() string {
	if version == "dev" {
		if bi, ok := debug.ReadBuildInfo(); ok && bi.Main.Version != "" && bi.Main.Version != "(devel)" {
			return bi.Main.Version
		}
	}
	return version
}

// ---------------------------------------------------------------------------
// per-input pipeline

func transcribeFile(ctx context.Context, cfg *config, jb job, apiKey string, keyterms []string) error {
	start := time.Now()
	logf := jb.logf
	live := !jb.multi && !cfg.quiet

	var raw []byte
	if isSavedResponse(jb.input) {
		var err error
		if raw, err = os.ReadFile(jb.input); err != nil {
			return fmt.Errorf("reading saved response: %w", err)
		}
		logf("▸ reusing saved Scribe response (no API call)")
	} else {
		src, err := buildAudioSource(ctx, jb.input)
		if err != nil {
			return err
		}
		logf("▸ audio: %s", src.desc)
		if raw, err = callScribe(ctx, cfg, src, apiKey, keyterms, logf, live); err != nil {
			return err
		}
		if cfg.rawPath != "" {
			if err := os.WriteFile(cfg.rawPath, raw, 0o644); err != nil {
				return fmt.Errorf("saving raw response: %w", err)
			}
		}
	}

	var res sttResponse
	if err := json.Unmarshal(raw, &res); err != nil {
		return fmt.Errorf("parsing Scribe response: %w", err)
	}
	paras := buildParagraphs(res.Words)
	if len(paras) == 0 {
		return errors.New("no speech found in input")
	}
	speakerIDs := speakerOrder(paras)

	var cues []segment
	if cfg.srt || cfg.vtt {
		cues = buildCues(res.Words)
	}

	names := map[string]string{}
	if !cfg.noLLM {
		names = nameAndCorrect(ctx, cfg, jb.name, res.LanguageCode, paras, cues, speakerIDs, logf)
	}

	display := displayNames(speakerIDs, names)
	total := paras[len(paras)-1].end
	if err := writeTranscript(jb.outPath, jb.name, res, paras, display, speakerIDs); err != nil {
		return err
	}
	writeSubs := func(ext string) error {
		p := strings.TrimSuffix(jb.outPath, ".txt") + ext
		if err := writeSubtitles(p, cues, display, ext == ".vtt"); err != nil {
			return err
		}
		logf("  also wrote %s", p)
		return nil
	}
	if cfg.srt {
		if err := writeSubs(".srt"); err != nil {
			return err
		}
	}
	if cfg.vtt {
		if err := writeSubs(".vtt"); err != nil {
			return err
		}
	}

	// segment text is single-space normalized, so spaces+1 is an exact word count
	words := 0
	for _, p := range paras {
		words += strings.Count(p.text, " ") + 1
	}
	logf("✓ %s — %s of audio, %d words, %d speakers, done in %s",
		jb.outPath, fmtClock(total, total), words, len(speakerIDs), time.Since(start).Round(time.Second))
	if cfg.quiet {
		fmt.Println(jb.outPath)
	}
	return nil
}

// nameAndCorrect runs the LLM pass and applies its speaker names and
// proper-noun corrections to the transcript (and subtitle cues) in place.
func nameAndCorrect(ctx context.Context, cfg *config, srcName, lang string, paras, cues []segment, speakerIDs []string, logf logFn) map[string]string {
	backend := pickLLMBackend()
	if backend == nil {
		logf("▸ no LLM available (claude CLI or ANTHROPIC_API_KEY/GEMINI_API_KEY) — keeping generic speaker labels")
		return nil
	}
	logf("▸ identifying speakers & checking proper nouns (%s)…", backend.name)
	start := time.Now()
	model := cfg.llmModel
	if model == "" {
		model = backend.defaultModel
	}
	result, err := backend.annotate(ctx, buildLLMPrompt(srcName, cfg.context, lang, paras), model)
	if err != nil {
		logf("  ! naming pass failed (%v) — keeping generic speaker labels", err)
		return nil
	}

	names := map[string]string{}
	for _, id := range speakerIDs {
		if name := cleanName(result.Speakers[id]); name != "" {
			names[id] = name
			logf("  · %s = %s", id, name)
		}
	}

	fixes := applyCorrections(result.Corrections, paras, cues)
	for _, f := range fixes {
		logf("  · %q → %q (%d×)", f.find, f.replace, f.count)
	}
	logf("  named %d/%d speakers, %d corrections in %s",
		len(names), len(speakerIDs), len(fixes), time.Since(start).Round(time.Second))
	return names
}

func outputPath(out, input string, multi bool) string {
	var dir, base string
	if input == "-" || isURL(input) {
		dir, base = ".", displayBase(input)+".txt"
	} else {
		dir = filepath.Dir(input)
		base = strings.TrimSuffix(filepath.Base(input), filepath.Ext(input)) + ".txt"
	}
	switch {
	case out == "":
		return filepath.Join(dir, base)
	case multi || isDir(out):
		return filepath.Join(out, base)
	default:
		return out
	}
}

func isDir(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}

func isURL(s string) bool {
	return strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://")
}

// isSavedResponse reports whether input is a raw Scribe response saved by
// --raw. Such inputs are reprocessed straight from disk — no second API call —
// so a different --context, --llm-model, --no-llm, or --srt/--vtt can be
// applied to an already-transcribed file for free.
func isSavedResponse(input string) bool {
	return strings.HasSuffix(strings.ToLower(input), ".json")
}

// displayBase is a short human name for an input: the file base name, "stdin",
// or a slug derived from a URL (YouTube video id, last path segment, or host).
func displayBase(input string) string {
	if input == "-" {
		return "stdin"
	}
	if !isURL(input) {
		return filepath.Base(input)
	}
	u, err := url.Parse(input)
	if err != nil {
		return "transcript"
	}
	if v := u.Query().Get("v"); v != "" { // youtube.com/watch?v=<id>
		return sanitizeName(v)
	}
	base := path.Base(u.Path)
	if base = strings.TrimSuffix(base, path.Ext(base)); base != "" && base != "." && base != "/" {
		return sanitizeName(base)
	}
	return sanitizeName(u.Hostname())
}

var unsafeNameChars = regexp.MustCompile(`[^\w.-]+`)

func sanitizeName(s string) string {
	s = unsafeNameChars.ReplaceAllString(s, "-")
	if s = strings.Trim(s, "-."); s == "" {
		return "transcript"
	}
	return s
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}

// ---------------------------------------------------------------------------
// audio sources: local files are extracted with ffmpeg and streamed into the
// upload; URLs are fetched server-side by ElevenLabs (source_url).

type audioSource struct {
	name      string // filename presented to the API (file uploads)
	desc      string // human-readable plan, for logging
	sourceURL string // when set, no upload happens
	oneShot   bool   // stdin: a consumed pipe cannot be replayed, so never retry
	open      func(ctx context.Context) (io.ReadCloser, error)
}

// Formats Scribe accepts as-is and that are already compact — uploaded directly.
var directExts = map[string]bool{
	".mp3": true, ".m4a": true, ".aac": true, ".ogg": true,
	".opus": true, ".flac": true, ".weba": true,
}

// codec → (output muxer, extension) for remux without re-encoding
var copyPlans = map[string][2]string{
	"aac":    {"adts", ".aac"},
	"mp3":    {"mp3", ".mp3"},
	"opus":   {"ogg", ".ogg"},
	"vorbis": {"ogg", ".ogg"},
	"flac":   {"flac", ".flac"},
}

func buildAudioSource(ctx context.Context, input string) (*audioSource, error) {
	if isURL(input) {
		return &audioSource{
			sourceURL: input,
			desc:      "remote URL, fetched server-side by ElevenLabs",
		}, nil
	}
	if input == "-" {
		// Scribe sniffs the format from content; oneShot disables retries.
		return &audioSource{
			name:    "audio",
			desc:    "stdin, streamed to upload",
			oneShot: true,
			open:    func(context.Context) (io.ReadCloser, error) { return io.NopCloser(os.Stdin), nil },
		}, nil
	}
	if _, err := os.Stat(input); err != nil {
		return nil, err
	}

	ext := strings.ToLower(filepath.Ext(input))
	if directExts[ext] {
		return &audioSource{
			name: "audio" + ext,
			desc: fmt.Sprintf("%s uploaded as-is", strings.TrimPrefix(ext, ".")),
			open: func(context.Context) (io.ReadCloser, error) { return os.Open(input) },
		}, nil
	}

	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return nil, fmt.Errorf("ffmpeg is required to extract audio from %s files (brew install ffmpeg)", ext)
	}
	codec, err := probeAudioCodec(ctx, input)
	if err != nil {
		return nil, err
	}

	var args []string
	var name, desc string
	if plan, ok := copyPlans[codec]; ok {
		args = []string{"-c:a", "copy", "-f", plan[0]}
		name, desc = "audio"+plan[1], fmt.Sprintf("%s track stream-copied (no re-encode)", codec)
	} else {
		// Unknown/uncompressed codec: lossless FLAC re-encode at native rate/channels.
		args = []string{"-c:a", "flac", "-f", "flac"}
		name, desc = "audio.flac", fmt.Sprintf("%s track re-encoded to FLAC (lossless)", codec)
	}

	return &audioSource{
		name: name,
		desc: desc + ", streamed to upload",
		open: func(ctx context.Context) (io.ReadCloser, error) {
			full := append([]string{"-hide_banner", "-nostdin", "-loglevel", "error",
				"-i", input, "-map", "0:a:0", "-vn"}, append(args, "pipe:1")...)
			cmd := exec.CommandContext(ctx, "ffmpeg", full...)
			var stderr bytes.Buffer
			cmd.Stderr = &stderr
			stdout, err := cmd.StdoutPipe()
			if err != nil {
				return nil, err
			}
			if err := cmd.Start(); err != nil {
				return nil, err
			}
			return &ffmpegReader{ReadCloser: stdout, cmd: cmd, stderr: &stderr}, nil
		},
	}, nil
}

type ffmpegReader struct {
	io.ReadCloser
	cmd    *exec.Cmd
	stderr *bytes.Buffer
}

func (f *ffmpegReader) Close() error {
	f.ReadCloser.Close()
	if err := f.cmd.Wait(); err != nil {
		return fmt.Errorf("ffmpeg: %v: %s", err, strings.TrimSpace(f.stderr.String()))
	}
	return nil
}

func probeAudioCodec(ctx context.Context, input string) (string, error) {
	out, err := exec.CommandContext(ctx, "ffprobe", "-v", "error", "-select_streams", "a:0",
		"-show_entries", "stream=codec_name", "-of", "csv=p=0", input).Output()
	if err != nil {
		return "", fmt.Errorf("ffprobe failed on %s: %v", input, err)
	}
	codec := strings.TrimSpace(string(out))
	if codec == "" {
		return "", errors.New("no audio stream found in file")
	}
	return codec, nil
}

// ---------------------------------------------------------------------------
// Scribe v2 API

type sttWord struct {
	Text      string  `json:"text"`
	Start     float64 `json:"start"`
	End       float64 `json:"end"`
	Type      string  `json:"type"` // word | spacing | audio_event
	SpeakerID string  `json:"speaker_id"`
}

type sttResponse struct {
	LanguageCode        string    `json:"language_code"`
	LanguageProbability float64   `json:"language_probability"`
	Words               []sttWord `json:"words"`
}

func callScribe(ctx context.Context, cfg *config, src *audioSource, apiKey string, keyterms []string, logf logFn, live bool) ([]byte, error) {
	// Clone the default transport so proxy support and dial/TLS defaults are
	// kept; only the header timeout changes (sync API: the response arrives
	// only when server-side processing is done).
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ResponseHeaderTimeout = 60 * time.Minute
	client := &http.Client{Transport: transport}

	attempts := 3
	if src.oneShot {
		attempts = 1
	}
	var lastErr error
	for attempt := 0; attempt < attempts; attempt++ {
		if attempt > 0 {
			logf("  attempt failed (%v) — retrying (%d/%d)…", lastErr, attempt, attempts-1)
			select {
			case <-time.After(time.Duration(attempt*attempt) * 3 * time.Second): // quadratic backoff: 3s, 12s
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		raw, retryable, err := scribeAttempt(ctx, client, cfg, src, apiKey, keyterms, logf, live)
		if err == nil {
			return raw, nil
		}
		lastErr = err
		if !retryable || ctx.Err() != nil {
			return nil, err
		}
	}
	return nil, lastErr
}

func scribeAttempt(ctx context.Context, client *http.Client, cfg *config, src *audioSource, apiKey string, keyterms []string, logf logFn, live bool) (raw []byte, retryable bool, err error) {
	var media io.ReadCloser
	if src.open != nil {
		if media, err = src.open(ctx); err != nil {
			return nil, false, err
		}
	}

	pr, pw := io.Pipe()
	mw := multipart.NewWriter(pw)
	go func() { // sole owner of media: closes it on every path
		err := func() error {
			fields := map[string]string{
				"model_id":               sttModel,
				"diarize":                "true",
				"timestamps_granularity": "word",
				"tag_audio_events":       "false",
				"no_verbatim":            fmt.Sprint(!cfg.verbatim),
			}
			if src.sourceURL != "" {
				fields["source_url"] = src.sourceURL
			}
			if cfg.language != "" {
				fields["language_code"] = cfg.language
			}
			if cfg.speakers > 0 {
				fields["num_speakers"] = fmt.Sprint(cfg.speakers)
			}
			for k, v := range fields {
				if err := mw.WriteField(k, v); err != nil {
					return err
				}
			}
			for _, term := range keyterms {
				if err := mw.WriteField("keyterms", term); err != nil {
					return err
				}
			}
			if media == nil {
				return nil
			}
			fw, err := mw.CreateFormFile("file", src.name)
			if err != nil {
				return err
			}
			_, err = io.Copy(fw, media)
			return err
		}()
		if media != nil {
			if cerr := media.Close(); err == nil {
				err = cerr // surfaces ffmpeg failures
			}
		}
		if err == nil {
			err = mw.Close()
		}
		pw.CloseWithError(err)
	}()

	prog := newProgress(logf, live, media != nil)
	defer prog.stop()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, sttEndpoint, prog.wrap(pr))
	if err != nil {
		return nil, false, err
	}
	req.Header.Set("xi-api-key", apiKey)
	req.Header.Set("Content-Type", mw.FormDataContentType())

	resp, err := client.Do(req)
	if err != nil {
		return nil, true, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()
	raw, err = io.ReadAll(resp.Body)
	if err != nil {
		return nil, true, fmt.Errorf("reading response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, resp.StatusCode == 429 || resp.StatusCode >= 500,
			fmt.Errorf("Scribe API returned %s: %s", resp.Status, truncate(strings.TrimSpace(string(raw)), 600))
	}
	prog.finish()
	return raw, false, nil
}

// progress reports request progress on stderr. On a TTY it live-updates a
// status line (upload bytes, then a transcribing heartbeat); otherwise it logs
// a heartbeat line every 30s so long waits are visible in piped output too.
type progress struct {
	logf      logFn
	tty       bool // live single-file run on a terminal
	hasUpload bool
	started   time.Time
	bytes     atomic.Int64
	upNano    atomic.Int64 // when the body finished uploading; 0 while uploading
	stopCh    chan struct{}
	stopOnce  sync.Once
}

func newProgress(logf logFn, live, hasUpload bool) *progress {
	p := &progress{
		logf:      logf,
		tty:       live && stderrIsTTY(),
		hasUpload: hasUpload,
		started:   time.Now(),
		stopCh:    make(chan struct{}),
	}
	go p.run()
	return p
}

const clearLine = "\r\033[K"

func (p *progress) wrap(r io.ReadCloser) io.ReadCloser { return &progressBody{p: p, r: r} }

func (p *progress) mb() float64 { return float64(p.bytes.Load()) / 1e6 }

func sinceNs(ns int64) time.Duration {
	return time.Since(time.Unix(0, ns)).Round(time.Second)
}

type progressBody struct {
	p *progress
	r io.ReadCloser
}

func (b *progressBody) Read(buf []byte) (int, error) {
	n, err := b.r.Read(buf)
	b.p.bytes.Add(int64(n))
	if err == io.EOF {
		b.p.upNano.CompareAndSwap(0, time.Now().UnixNano())
	}
	return n, err
}

// Close reaches the underlying pipe reader, so when the HTTP transport closes
// the request body (including on early errors) the producer goroutine — and
// any ffmpeg behind it — unblocks and exits instead of being stranded.
func (b *progressBody) Close() error { return b.r.Close() }

func (p *progress) run() {
	interval := 30 * time.Second
	if p.tty {
		interval = 500 * time.Millisecond
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-p.stopCh:
			return
		case <-t.C:
			up := p.upNano.Load()
			switch {
			case p.tty && up == 0:
				fmt.Fprintf(os.Stderr, clearLine+"  ↑ %.1f MB", p.mb())
			case p.tty:
				fmt.Fprintf(os.Stderr, clearLine+"  transcribing… %s", sinceNs(up))
			case up == 0:
				p.logf("  ↑ %.1f MB uploaded so far…", p.mb())
			default:
				p.logf("  transcribing… %s elapsed", sinceNs(up))
			}
		}
	}
}

func (p *progress) stop() {
	p.stopOnce.Do(func() {
		close(p.stopCh)
		if p.tty {
			fmt.Fprint(os.Stderr, clearLine)
		}
	})
}

func (p *progress) finish() {
	p.stop()
	up := p.upNano.Load()
	if up == 0 {
		return
	}
	if p.hasUpload {
		p.logf("  ↑ %.1f MB uploaded in %s · transcribed in %s",
			p.mb(), time.Unix(0, up).Sub(p.started).Round(time.Second), sinceNs(up))
	} else {
		p.logf("  transcribed in %s", sinceNs(up))
	}
}

func stderrIsTTY() bool {
	fi, err := os.Stderr.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

// ---------------------------------------------------------------------------
// transcript assembly: one word-stream segmenter, two break policies
// (paragraphs for the .txt transcript, cues for subtitles)

type segment struct {
	speaker string // raw speaker id, e.g. "speaker_0"
	start   float64
	end     float64
	text    string
}

// buildSegments walks the word stream and groups it into speaker-attributed
// segments. A new segment always starts on speaker change; brk may request
// additional breaks based on the segment so far and the upcoming word.
func buildSegments(words []sttWord, brk func(cur *segment, text string, w sttWord) bool) []segment {
	var out []segment
	var sb strings.Builder
	var cur segment
	active := false

	flush := func() {
		if text := strings.Join(strings.Fields(sb.String()), " "); text != "" {
			cur.text = text
			out = append(out, cur)
		}
		sb.Reset()
		active = false
	}

	for _, w := range words {
		switch w.Type {
		case "audio_event":
			// tag_audio_events is always off, but skip any event the API
			// returns anyway rather than splicing "(laughter)" into text.
			continue
		case "spacing":
			if active {
				sb.WriteString(" ")
			}
			continue
		}
		if active && (w.SpeakerID != cur.speaker || brk(&cur, sb.String(), w)) {
			flush()
		}
		if !active {
			cur = segment{speaker: w.SpeakerID, start: w.Start}
			active = true
		}
		sb.WriteString(w.Text)
		cur.end = w.End
	}
	flush()
	return out
}

func buildParagraphs(words []sttWord) []segment {
	return buildSegments(words, func(cur *segment, text string, w sttWord) bool {
		return w.Start-cur.start > maxTurnSeconds && endsSentence(text)
	})
}

func endsSentence(s string) bool {
	s = strings.TrimRight(s, " \t\"'”’)")
	r, _ := utf8.DecodeLastRuneInString(s) // RuneError for empty s: not a match
	return strings.ContainsRune(".!?…。！？", r)
}

func speakerOrder(paras []segment) []string {
	seen := map[string]bool{}
	var order []string
	for _, p := range paras {
		if !seen[p.speaker] {
			seen[p.speaker] = true
			order = append(order, p.speaker)
		}
	}
	return order
}

// displayNames maps raw speaker ids to final labels: LLM-provided names, or
// "Speaker N" by order of first appearance.
func displayNames(order []string, names map[string]string) map[string]string {
	out := map[string]string{}
	for i, id := range order {
		if n := names[id]; n != "" {
			out[id] = n
		} else {
			out[id] = fmt.Sprintf("Speaker %d", i+1)
		}
	}
	return out
}

func writeTranscript(path, srcName string, res sttResponse, paras []segment, display map[string]string, order []string) error {
	total := paras[len(paras)-1].end
	var b bytes.Buffer
	labels := make([]string, len(order))
	for i, id := range order {
		labels[i] = display[id]
	}
	fmt.Fprintf(&b, "# %s\n", srcName)
	fmt.Fprintf(&b, "# Transcribed %s · ElevenLabs %s · language %s (%.0f%% confidence)\n",
		time.Now().Format("2006-01-02"), sttModel, res.LanguageCode, res.LanguageProbability*100)
	fmt.Fprintf(&b, "# Duration %s · %d speakers: %s\n\n",
		fmtClock(total, total), len(order), strings.Join(labels, ", "))

	for _, p := range paras {
		fmt.Fprintf(&b, "[%s] %s: %s\n\n", fmtClock(p.start, total), display[p.speaker], p.text)
	}
	return os.WriteFile(path, b.Bytes(), 0o644)
}

// fmtClock renders seconds as MM:SS, or HH:MM:SS when the recording is >= 1h.
func fmtClock(sec, total float64) string {
	s := int(sec + 0.5)
	if total >= 3600 {
		return fmt.Sprintf("%d:%02d:%02d", s/3600, (s%3600)/60, s%60)
	}
	return fmt.Sprintf("%02d:%02d", s/60, s%60)
}

// ---------------------------------------------------------------------------
// subtitle cues (--srt / --vtt)

const (
	maxCueChars  = 84  // ~2 lines of 42
	maxCueSec    = 5.0 // hard cap on cue duration
	cueBreakSec  = 2.5 // prefer breaking after this, at sentence ends
	cueGapSec    = 1.5 // silence gap that forces a new cue
	cueWrapChars = 42
)

func buildCues(words []sttWord) []segment {
	return buildSegments(words, func(cur *segment, text string, w sttWord) bool {
		dur := w.Start - cur.start
		return w.Start-cur.end >= cueGapSec ||
			dur >= maxCueSec ||
			len(text)+len(w.Text) > maxCueChars ||
			(dur >= cueBreakSec && endsSentence(text))
	})
}

func writeSubtitles(path string, cues []segment, display map[string]string, vtt bool) error {
	var b bytes.Buffer
	sep := ","
	if vtt {
		b.WriteString("WEBVTT\n\n")
		sep = "."
	}
	prevSpeaker := ""
	for i, c := range cues {
		if !vtt {
			fmt.Fprintf(&b, "%d\n", i+1)
		}
		fmt.Fprintf(&b, "%s --> %s\n", cueTime(c.start, sep), cueTime(c.end, sep))
		text := c.text
		if c.speaker != prevSpeaker {
			text = display[c.speaker] + ": " + text
			prevSpeaker = c.speaker
		}
		b.WriteString(wrapCue(text) + "\n\n")
	}
	return os.WriteFile(path, b.Bytes(), 0o644)
}

// wrapCue breaks a cue into two lines at the space nearest its midpoint.
func wrapCue(s string) string {
	if len(s) <= cueWrapChars {
		return s
	}
	mid, best := len(s)/2, -1
	for i, r := range s {
		if r == ' ' && (best < 0 || abs(i-mid) < abs(best-mid)) {
			best = i
		}
	}
	if best <= 0 {
		return s
	}
	return s[:best] + "\n" + s[best+1:]
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

func cueTime(sec float64, sep string) string {
	ms := int(sec*1000 + 0.5)
	return fmt.Sprintf("%02d:%02d:%02d%s%03d", ms/3600000, ms/60000%60, ms/1000%60, sep, ms%1000)
}

// ---------------------------------------------------------------------------
// keyterm mining: capitalized runs from the context prompt bias recognition

// keytermPattern captures runs of capitalized words (any script — \p{Lu}, not
// just A-Z). RE2 has no lookbehind, so the leading non-letter is matched
// explicitly and the term is submatch 1.
var keytermPattern = regexp.MustCompile(`(?:^|[^\p{L}\p{N}])(\p{Lu}[\p{L}\p{N}'’.-]*(?: \p{Lu}[\p{L}\p{N}'’.-]*){0,4})`)

func collectKeyterms(explicit, contextText string) []string {
	seen := map[string]bool{}
	var out []string
	add := func(t string) {
		t = strings.Trim(strings.TrimSpace(t), `<>{}[]\`)
		n := utf8.RuneCountInString(t) // the API's 50-char limit counts characters
		if n < 2 || n >= 50 || seen[strings.ToLower(t)] || len(out) >= 1000 {
			return
		}
		seen[strings.ToLower(t)] = true
		out = append(out, t)
	}
	for _, t := range strings.Split(explicit, ",") {
		if t != "" {
			add(t)
		}
	}
	for _, m := range keytermPattern.FindAllStringSubmatch(contextText, -1) {
		add(m[1])
	}
	return out
}

// ---------------------------------------------------------------------------
// LLM naming pass: tiny JSON out (speaker names + proper-noun corrections)

type correction struct {
	Find    string `json:"find"`
	Replace string `json:"replace"`
}

type llmResult struct {
	Speakers    map[string]string `json:"speakers"`
	Corrections []correction      `json:"corrections"`
}

func buildLLMPrompt(fileName, userContext, lang string, paras []segment) string {
	var t strings.Builder
	for _, p := range paras {
		fmt.Fprintf(&t, "%s: %s\n", p.speaker, p.text)
	}
	transcript := t.String()
	if len(transcript) > maxPromptChars {
		transcript = transcript[:maxPromptChars*3/4] +
			"\n[... middle of transcript omitted ...]\n" +
			transcript[len(transcript)-maxPromptChars/4:]
	}
	if userContext == "" {
		userContext = "(none provided)"
	}
	return fmt.Sprintf(`You are annotating a machine-generated, diarized transcript. Speakers carry generic ids (speaker_0, speaker_1, ...).

Task 1 — name the speakers. Use introductions, how speakers address each other, the file name, and the user context. Only assign a name when the evidence is clear; otherwise use null. Never guess.

Task 2 — list corrections for clearly misrecognized proper nouns or technical terms (people, organizations, places, products, titles). "find" must be the exact, case-sensitive text as it appears in the transcript and must not be a common English word. Only include corrections you are confident about.

Reply with ONLY this JSON, no prose, no code fences:
{"speakers": {"speaker_0": "Full Name or null", ...}, "corrections": [{"find": "...", "replace": "..."}]}

FILE NAME: %s
LANGUAGE: %s
USER CONTEXT: %s

TRANSCRIPT:
%s`, fileName, lang, userContext, transcript)
}

type llmBackend struct {
	name         string
	defaultModel string
	annotate     func(ctx context.Context, prompt, model string) (*llmResult, error)
}

// pickLLMBackend resolves once per process; env and PATH don't change mid-run.
// Preference: API keys first (OpenAI, Anthropic, Gemini), then a local CLI
// (Codex, Claude) if installed.
var pickLLMBackend = sync.OnceValue(func() *llmBackend {
	if os.Getenv("OPENAI_API_KEY") != "" {
		return &llmBackend{"openai api", defaultOpenAIModel, callOpenAI}
	}
	if os.Getenv("ANTHROPIC_API_KEY") != "" {
		return &llmBackend{"anthropic api", defaultClaudeModel, callAnthropic}
	}
	if geminiKey() != "" {
		return &llmBackend{"gemini api", defaultGeminiModel, callGemini}
	}
	if _, err := exec.LookPath("codex"); err == nil {
		return &llmBackend{"codex cli", defaultCodexModel, callCodexCLI}
	}
	if _, err := exec.LookPath("claude"); err == nil {
		return &llmBackend{"claude cli", defaultClaudeModel, callClaudeCLI}
	}
	return nil
})

func geminiKey() string {
	return firstEnv("GEMINI_API_KEY", "GOOGLE_API_KEY")
}

func callClaudeCLI(ctx context.Context, prompt, model string) (*llmResult, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "claude", "-p", "--model", model)
	cmd.Dir = os.TempDir() // avoid loading any project context
	cmd.Stdin = strings.NewReader(prompt)
	var out, errBuf bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errBuf
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("claude cli: %v: %s", err, strings.TrimSpace(errBuf.String()))
	}
	return parseLLMJSON(out.String())
}

// callCodexCLI drives `codex exec`, whose stdout is a verbose session log; the
// clean final message is captured via --output-last-message. stdin is closed so
// codex doesn't block waiting for extra input.
func callCodexCLI(ctx context.Context, prompt, model string) (*llmResult, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	out, err := os.CreateTemp("", "transcribe-codex-*.txt")
	if err != nil {
		return nil, err
	}
	defer os.Remove(out.Name())
	out.Close()
	cmd := exec.CommandContext(ctx, "codex", "exec",
		"--skip-git-repo-check", "--sandbox", "read-only",
		"-m", model, "--output-last-message", out.Name(), prompt)
	cmd.Dir = os.TempDir()
	cmd.Stdin = strings.NewReader("")
	var errBuf bytes.Buffer
	cmd.Stdout, cmd.Stderr = io.Discard, &errBuf
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("codex cli: %v: %s", err, truncate(strings.TrimSpace(errBuf.String()), 200))
	}
	data, err := os.ReadFile(out.Name())
	if err != nil {
		return nil, err
	}
	return parseLLMJSON(string(data))
}

func callOpenAI(ctx context.Context, prompt, model string) (*llmResult, error) {
	body, _ := json.Marshal(map[string]any{
		"model":    model,
		"messages": []map[string]string{{"role": "user", "content": prompt}},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("authorization", "Bearer "+os.Getenv("OPENAI_API_KEY"))
	req.Header.Set("content-type", "application/json")
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := doJSON(req, &parsed); err != nil {
		return nil, err
	}
	var text strings.Builder
	for _, c := range parsed.Choices {
		text.WriteString(c.Message.Content)
	}
	return parseLLMJSON(text.String())
}

func callAnthropic(ctx context.Context, prompt, model string) (*llmResult, error) {
	body, _ := json.Marshal(map[string]any{
		"model":      model,
		"max_tokens": 4000,
		"messages":   []map[string]string{{"role": "user", "content": prompt}},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-api-key", os.Getenv("ANTHROPIC_API_KEY"))
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("content-type", "application/json")
	var parsed struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := doJSON(req, &parsed); err != nil {
		return nil, err
	}
	var text strings.Builder
	for _, c := range parsed.Content {
		text.WriteString(c.Text)
	}
	return parseLLMJSON(text.String())
}

func callGemini(ctx context.Context, prompt, model string) (*llmResult, error) {
	body, _ := json.Marshal(map[string]any{
		"contents": []map[string]any{{"parts": []map[string]string{{"text": prompt}}}},
	})
	u := "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-goog-api-key", geminiKey())
	req.Header.Set("content-type", "application/json")
	var parsed struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	if err := doJSON(req, &parsed); err != nil {
		return nil, err
	}
	var text strings.Builder
	for _, c := range parsed.Candidates {
		for _, p := range c.Content.Parts {
			text.WriteString(p.Text)
		}
	}
	return parseLLMJSON(text.String())
}

func doJSON(req *http.Request, out any) error {
	resp, err := (&http.Client{Timeout: 5 * time.Minute}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: %s", resp.Status, truncate(strings.TrimSpace(string(raw)), 400))
	}
	return json.Unmarshal(raw, out)
}

func parseLLMJSON(s string) (*llmResult, error) {
	i, j := strings.Index(s, "{"), strings.LastIndex(s, "}")
	if i < 0 || j <= i {
		return nil, fmt.Errorf("no JSON in LLM reply: %s", truncate(s, 120))
	}
	var res llmResult
	if err := json.Unmarshal([]byte(s[i:j+1]), &res); err != nil {
		return nil, fmt.Errorf("bad JSON from LLM: %w", err)
	}
	return &res, nil
}

// cleanName returns a usable speaker name from the LLM, or "" to keep the
// generic label.
func cleanName(name string) string {
	name = strings.TrimSpace(name)
	switch lower := strings.ToLower(name); {
	case name == "", lower == "null", lower == "unknown",
		len(name) > 60, strings.ContainsAny(name, "\n{}"):
		return ""
	}
	return name
}

type appliedFix struct {
	find, replace string
	count         int
}

// applyCorrections rewrites confidently-misrecognized proper nouns in place.
// Only exact, boundary-respecting, case-sensitive matches are replaced.
func applyCorrections(corrections []correction, paras, cues []segment) []appliedFix {
	var applied []appliedFix
	for i, c := range corrections {
		if i >= 100 || len(c.Find) < 3 || c.Find == c.Replace || c.Replace == "" ||
			strings.Count(c.Find, " ") > 5 ||
			len(c.Replace) > 60 || strings.ContainsAny(c.Replace, "\n{}") {
			continue
		}
		// RE2's \b is ASCII-defined, so it is only added when the edge rune is
		// an ASCII word character; non-ASCII edges match without a boundary.
		pat := regexp.QuoteMeta(c.Find) // QuoteMeta guarantees a valid pattern
		if first, _ := utf8.DecodeRuneInString(c.Find); asciiWordChar(first) {
			pat = `\b` + pat
		}
		if last, _ := utf8.DecodeLastRuneInString(c.Find); asciiWordChar(last) {
			pat += `\b`
		}
		re := regexp.MustCompile(pat)
		count := 0
		for _, list := range [][]segment{paras, cues} {
			for j := range list {
				// replacement is returned from a func, so it is always literal
				list[j].text = re.ReplaceAllStringFunc(list[j].text, func(string) string {
					count++
					return c.Replace
				})
			}
		}
		if count > 0 {
			applied = append(applied, appliedFix{c.Find, c.Replace, count})
		}
	}
	return applied
}

func asciiWordChar(r rune) bool {
	return r == '_' || r >= '0' && r <= '9' || r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z'
}

// ---------------------------------------------------------------------------
// env helpers

func loadDotEnv(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "export "))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		k, v = strings.TrimSpace(k), strings.Trim(strings.TrimSpace(v), `"'`)
		if os.Getenv(k) == "" {
			os.Setenv(k, v)
		}
	}
}

func firstEnv(keys ...string) string {
	for _, k := range keys {
		if v := os.Getenv(k); v != "" {
			return v
		}
	}
	return ""
}
