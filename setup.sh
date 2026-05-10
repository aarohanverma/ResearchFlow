#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ResearchFlow — Setup & Run script
#
# Usage:
#   ./setup.sh            auto: runs setup if fresh, starts servers if done
#   ./setup.sh --setup    force full setup (reinstall deps, re-seed)
#   ./setup.sh --run      skip setup, just start servers
#   ./setup.sh --reset    wipe saved state and re-run full setup
#
# Setup state is saved to .setup_state after a successful setup.
# On subsequent runs the script reads it and goes straight to launch.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

info()    { echo -e "${CYAN}ℹ  $*${RESET}"; }
success() { echo -e "${GREEN}✓  $*${RESET}"; }
warn()    { echo -e "${YELLOW}⚠  $*${RESET}"; }
error()   { echo -e "${RED}✗  $*${RESET}"; }
step()    { echo -e "\n${BOLD}${CYAN}┌─  $*${RESET}\n"; }
die()     { error "$*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE="$SCRIPT_DIR/.env.local"
STATE_FILE="$SCRIPT_DIR/.setup_state"
VENV_DIR="$SCRIPT_DIR/backend/.venv"

# ── Parse arguments ───────────────────────────────────────────────────────────
MODE="auto"
for arg in "${@:-}"; do
    case "$arg" in
        --setup) MODE="setup" ;;
        --run)   MODE="run"   ;;
        --reset)
            rm -f "$STATE_FILE"
            info "Setup state cleared — running full setup."
            MODE="setup"
            ;;
        --help|-h)
            echo "Usage: ./setup.sh [--setup | --run | --reset]"
            echo "  (no args)  auto-detect: setup if fresh, start if done"
            echo "  --setup    force full setup"
            echo "  --run      skip setup, start servers immediately"
            echo "  --reset    wipe .setup_state and run full setup"
            exit 0
            ;;
    esac
done

# ── State file helpers ────────────────────────────────────────────────────────
# Read a value from the state file
read_state() {
    local key="$1"
    grep -E "^${key}=" "$STATE_FILE" 2>/dev/null | head -1 | cut -d= -f2- || true
}

# Write the setup state file
write_state() {
    cat > "$STATE_FILE" << STEOF
# ResearchFlow setup state — do not edit manually
SETUP_VERSION=2
SETUP_DATE=$(date '+%Y-%m-%d %H:%M:%S')
PYTHON_CMD=${PYTHON_CMD}
PYTHON_VER=${PY_VER}
VENV_DIR=${VENV_DIR}
DC=${DC}
STEOF
    success "Setup state saved to .setup_state"
}

# ── Env file helpers ──────────────────────────────────────────────────────────
# grep returns exit 1 when no match — || true prevents set -e from killing the script
read_env_val() {
    local key="$1" file="${2:-$ENV_FILE}"
    { grep -E "^${key}=" "$file" 2>/dev/null || true; } | head -1 | cut -d= -f2- | sed "s/^['\"]//;s/['\"]$//"
}

is_real_key() {
    local val="$1"
    [[ -z "$val" ]] && return 1
    local -a bad=("your-" "-your-" "change-me" "placeholder" "example"
                  "sk-your" "AIza-your" "sk-ant-your" "re_your" "ls__your"
                  "key-here" "api-key")
    for p in "${bad[@]}"; do
        [[ "$val" == *"$p"* ]] && return 1
    done
    return 0
}

mask() { local v="$1"; [[ ${#v} -gt 10 ]] && echo "${v:0:10}…" || echo "****"; }

# collect_key BASH_VAR "Label" "hint" [ENV_FILE_KEY]
# ENV_FILE_KEY defaults to BASH_VAR when the names differ (e.g. OPENAI_KEY vs OPENAI_API_KEY)
collect_key() {
    local varname="$1" label="$2" hint="$3" envkey="${4:-$1}"
    local current
    current="$(read_env_val "$envkey")"
    if is_real_key "$current"; then
        printf "  ${GREEN}✓${RESET}  %-22s already set  ${DIM}($(mask "$current"))${RESET}\n" "$label"
        printf -v "$varname" '%s' "$current"
    else
        printf "  %-22s  " "$label"
        local val
        read -r val || true          # read returns 1 on EOF; guard against set -e
        printf -v "$varname" '%s' "$val"
    fi
}

# ── Auto-detect mode ──────────────────────────────────────────────────────────
if [[ "$MODE" == "auto" ]]; then
    if [[ -f "$STATE_FILE" && -f "$ENV_FILE" && -d "$VENV_DIR" ]]; then
        MODE="run"
    else
        MODE="setup"
    fi
fi

# ── Docker compose detection (needed in both modes) ───────────────────────────
detect_dc() {
    if docker compose version &>/dev/null 2>&1; then
        DC="docker compose"
    elif command -v docker-compose &>/dev/null; then
        DC="docker-compose"
    else
        die "Neither 'docker compose' nor 'docker-compose' found. Please update Docker."
    fi
}

# ── Start DB + wait for ready ─────────────────────────────────────────────────
start_db() {
    info "Starting database via: $DC up db -d"
    $DC up db -d

    info "Waiting for PostgreSQL to accept connections …"
    local ready=0
    for i in $(seq 1 30); do
        if $DC exec -T db pg_isready -U researchflow -q 2>/dev/null; then
            ready=1; break
        fi
        printf "  ${DIM}attempt %d/30 …${RESET}\r" "$i"
        sleep 2
    done
    printf "\n"
    if [[ $ready -eq 0 ]]; then
        error "PostgreSQL did not become ready within 60 seconds."
        echo -e "  Diagnose with: ${CYAN}$DC logs db${RESET}"
        exit 1
    fi
    success "PostgreSQL is ready"
}

# ── Launch both servers ───────────────────────────────────────────────────────
launch_servers() {
    local venv_dir="$1"

    # Kill any processes occupying the ports before starting
    for port in 8000 3000; do
        local pids
        pids="$(lsof -ti ":$port" 2>/dev/null || true)"
        if [[ -n "$pids" ]]; then
            warn "Port $port in use — killing existing process(es): $pids"
            kill -9 $pids 2>/dev/null || true
            sleep 0.5
        fi
    done

    # shellcheck source=/dev/null
    source "$venv_dir/bin/activate"

    info "Starting backend on :8000 …"
    cd "$SCRIPT_DIR/backend"
    uvicorn main:app --reload --port 8000 &
    BACKEND_PID=$!
    cd "$SCRIPT_DIR"

    sleep 3

    if curl -sf http://localhost:8000/health &>/dev/null; then
        success "Backend healthy  →  http://localhost:8000"
    else
        warn "Backend may still be starting — check http://localhost:8000/health"
    fi

    info "Building and starting frontend in production mode on :3000 …"
    cd "$SCRIPT_DIR/frontend"
    npm run build && npm run start &
    FRONTEND_PID=$!
    cd "$SCRIPT_DIR"

    echo ""
    success "Both servers running!"
    echo ""
    echo -e "    ${BOLD}Frontend:${RESET}   http://localhost:3000"
    echo -e "    ${BOLD}Backend:${RESET}    http://localhost:8000/docs"
    echo -e "    ${BOLD}Debug:${RESET}      http://localhost:8000/debug/status"
    echo ""
    echo -e "  ${DIM}Login:  test@researchflow.ai  /  ResearchFlow2024!${RESET}"
    echo ""
    echo -e "  ${DIM}Press Ctrl+C to stop both servers.${RESET}"
    echo ""

    cleanup() {
        echo ""
        info "Stopping servers …"
        kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
        success "Servers stopped. Goodbye!"
    }
    trap cleanup INT TERM
    wait
}

# ═════════════════════════════════════════════════════════════════════════════
# RUN MODE — setup already done, just start everything
# ═════════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "run" ]]; then
    echo ""
    echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${GREEN}║       ResearchFlow — Starting                        ║${RESET}"
    echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════╝${RESET}"
    echo ""
    echo -e "  ${DIM}Setup already complete. Loading saved config …${RESET}"
    echo -e "  ${DIM}Run './setup.sh --setup' to redo setup or '--reset' to wipe state.${RESET}"
    echo ""

    # Load saved state
    SAVED_VENV="$(read_state VENV_DIR)"
    SAVED_DC="$(read_state DC)"
    SAVED_PYTHON_VER="$(read_state PYTHON_VER)"
    SETUP_DATE="$(read_state SETUP_DATE)"

    # Validate state
    if [[ -z "$SAVED_VENV" || ! -d "$SAVED_VENV" ]]; then
        warn "Saved venv not found at '$SAVED_VENV' — falling back to $VENV_DIR"
        SAVED_VENV="$VENV_DIR"
    fi
    if [[ -z "$SAVED_DC" ]]; then
        detect_dc
        SAVED_DC="$DC"
    fi
    DC="$SAVED_DC"

    success "Config loaded  (setup: $SETUP_DATE, Python $SAVED_PYTHON_VER)"
    echo ""

    # Check docker is up
    if ! docker info &>/dev/null 2>&1; then
        die "Docker daemon is not running. Please start Docker."
    fi

    start_db

    echo ""
    read -rp "  Launch backend + frontend? [Y/n] " launch
    echo ""
    if [[ ! "${launch,,}" =~ ^n ]]; then
        launch_servers "$SAVED_VENV"
    else
        echo ""
        info "To start manually:"
        echo -e "    cd backend && source .venv/bin/activate"
        echo -e "    uvicorn main:app --reload --port 8000"
        echo ""
        echo -e "    cd frontend && npm run dev"
    fi

    exit 0
fi

# ═════════════════════════════════════════════════════════════════════════════
# SETUP MODE
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║       ResearchFlow — First-Time Setup                ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${DIM}Estimated time: 2–5 min (first run), <30 s (re-run)${RESET}"
echo -e "  ${DIM}API keys already in .env.local are never overwritten.${RESET}"
echo ""

# ── [1/7] Prerequisites ───────────────────────────────────────────────────────
step "[1/7] Checking prerequisites"

require_cmd() {
    local cmd="$1" label="$2" hint="$3"
    if command -v "$cmd" &>/dev/null; then
        success "$label  →  $(command -v "$cmd")"
    else
        error "$label not found."
        echo -e "       ${DIM}Install: $hint${RESET}"
        exit 1
    fi
}

# ── System packages (Ubuntu/Debian) ───────────────────────────────────────────
# Install essential non-Python dependencies in one shot to minimise sudo prompts.
_APT_PKGS=()
command -v ffmpeg          &>/dev/null || _APT_PKGS+=("ffmpeg")
command -v chromium-browser &>/dev/null || command -v chromium &>/dev/null || _APT_PKGS+=("chromium-browser")

if [[ ${#_APT_PKGS[@]} -gt 0 ]]; then
    info "Installing system packages: ${_APT_PKGS[*]}"
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y "${_APT_PKGS[@]}" 2>/dev/null && \
            success "System packages installed: ${_APT_PKGS[*]}" || \
            warn "apt-get failed — install manually: sudo apt-get install -y ${_APT_PKGS[*]}"
    elif command -v brew &>/dev/null; then
        # macOS
        for pkg in "${_APT_PKGS[@]}"; do
            [[ "$pkg" == "chromium-browser" ]] && pkg="chromium"
            brew install "$pkg" 2>/dev/null || warn "brew install $pkg failed"
        done
    else
        warn "Package manager not found. Please install manually: ${_APT_PKGS[*]}"
    fi
fi

require_cmd docker  "Docker " "https://docs.docker.com/get-docker/"
require_cmd python3 "Python3" "https://python.org  (need 3.10+)"
require_cmd node    "Node.js" "https://nodejs.org  (need 20+)"
require_cmd npm     "npm    " "bundled with Node.js"

# Python: prefer 3.11, accept 3.10
PYTHON_CMD="python3"
command -v python3.11 &>/dev/null && PYTHON_CMD="python3.11"

PYENV_BIN="${PYENV_ROOT:-$HOME/.pyenv}/bin/pyenv"
if [[ -x "$PYENV_BIN" ]] || command -v pyenv &>/dev/null; then
    PYENV_CMD="pyenv"
    [[ -x "$PYENV_BIN" ]] && PYENV_CMD="$PYENV_BIN"

    PY311_VER=$("$PYENV_CMD" versions --bare 2>/dev/null | grep -E "^3\.11\." | tail -1 || true)

    if [[ -n "$PY311_VER" ]]; then
        PYTHON_CMD="${PYENV_ROOT:-$HOME/.pyenv}/versions/${PY311_VER}/bin/python3"
        success "pyenv Python $PY311_VER found — will use for virtualenv"
    else
        echo ""
        warn "Python 3.11 not found in pyenv (installed: $("$PYENV_CMD" versions --bare | tr '\n' ' '))"
        read -rp "  Install Python 3.11 via pyenv now? (~3–5 min) [Y/n] " install_py
        if [[ ! "${install_py,,}" =~ ^n ]]; then
            info "Installing Python 3.11.9 via pyenv …"
            "$PYENV_CMD" install 3.11.9 --skip-existing
            PYTHON_CMD="${PYENV_ROOT:-$HOME/.pyenv}/versions/3.11.9/bin/python3"
            success "Python 3.11.9 installed"
        else
            info "Skipping — using current Python"
        fi
    fi
fi

PY_VER=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 || ("$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10) ]]; then
    die "Python 3.10+ required — found $PY_VER."
fi
[[ "$PY_MINOR" -lt 11 ]] \
    && warn "Using Python $PY_VER (3.11+ recommended)" \
    || success "Python $PY_VER — OK"

NODE_VER=$(node --version | tr -d 'v')
NODE_MAJOR=$(echo "$NODE_VER" | cut -d. -f1)
[[ "$NODE_MAJOR" -lt 20 ]] \
    && warn "Node $NODE_VER (20+ recommended, continuing)" \
    || success "Node $NODE_VER — OK"

if ! docker info &>/dev/null 2>&1; then
    die "Docker daemon not running. Start Docker Desktop or: sudo systemctl start docker"
fi
success "Docker daemon — running"

# ── Optional CLIs for media generation (slides + podcast) ─────────────────────
# Slide rendering uses @marp-team/marp-cli; absence falls back to raw markdown.
# Podcast TTS does not strictly need ffmpeg (OpenAI MP3 segments concat as bytes),
# but ffmpeg is recommended for any future audio post-processing.
if command -v marp &>/dev/null; then
    success "marp-cli — available (slide rendering enabled)"
elif command -v npx &>/dev/null; then
    success "npx available — marp-cli will be downloaded on first slide render"
else
    warn "marp-cli not found — slide generation will return raw Markdown."
    echo -e "       ${DIM}Install: npm install -g @marp-team/marp-cli${RESET}"
fi

# ── [2/7] API keys ────────────────────────────────────────────────────────────
step "[2/7] API key configuration"

if [[ ! -f "$ENV_FILE" ]]; then
    [[ -f "$SCRIPT_DIR/.env.example" ]] && cp "$SCRIPT_DIR/.env.example" "$ENV_FILE" || touch "$ENV_FILE"
    info "Created .env.local"
fi

echo -e "  ${DIM}Keys already in .env.local are reused. Press Enter to skip any key.${RESET}"
echo ""
echo -e "  ${BOLD}Required (at least OpenAI or Google):${RESET}"
collect_key OPENAI_KEY    "OpenAI API key"  "sk-proj-..."  OPENAI_API_KEY
collect_key GOOGLE_KEY    "Google AI key"   "AIza..."      GOOGLE_API_KEY
echo ""
echo -e "  ${BOLD}Optional:${RESET}"
collect_key ANTHROPIC_KEY "Anthropic key"   "sk-ant-..."   ANTHROPIC_API_KEY
collect_key RESEND_KEY    "Resend key"      "re_..."       RESEND_API_KEY
collect_key LANGSMITH_KEY "LangSmith key"   "ls__..."      LANGSMITH_API_KEY
echo ""

if ! is_real_key "$OPENAI_KEY" && ! is_real_key "$GOOGLE_KEY"; then
    warn "No LLM key — Study, Chat, and Genie won't work until you add one."
    echo -e "  ${DIM}Add keys later via Settings → Provider Config, or edit .env.local.${RESET}"
else
    is_real_key "$OPENAI_KEY" && success "LLM provider: OpenAI"
    is_real_key "$GOOGLE_KEY" && success "Embedding provider: Gemini"
fi

if is_real_key "$GOOGLE_KEY"; then
    EMB_PROVIDER="gemini"; EMB_MODEL="gemini-embedding-2-preview"; EMB_DIM="768"
else
    EMB_PROVIDER="openai"; EMB_MODEL="text-embedding-3-large"; EMB_DIM="3072"
fi

EXISTING_JWT="$(read_env_val JWT_SECRET)"
if is_real_key "$EXISTING_JWT" && [[ "$EXISTING_JWT" != "change-me"* && "$EXISTING_JWT" != "local-dev" ]]; then
    JWT_VAL="$EXISTING_JWT"
else
    JWT_VAL="local-dev-$(openssl rand -hex 16 2>/dev/null || "$PYTHON_CMD" -c 'import secrets; print(secrets.token_hex(16))')"
fi

# ── [3/7] Write .env.local ────────────────────────────────────────────────────
step "[3/7] Writing .env.local"

TMP_ENV="$(mktemp)"
cat > "$TMP_ENV" << ENVEOF
# ─── ResearchFlow — Local Development Environment ──────────────────────────────
# Generated by setup.sh on $(date '+%Y-%m-%d %H:%M:%S')
# DO NOT commit this file — it is listed in .gitignore.

DATABASE_URL=postgresql+asyncpg://researchflow:researchflow@localhost:5432/researchflow

CACHE_BACKEND=local
CACHE_DIR=${HOME}/.cache/researchflow
REDIS_URL=redis://localhost:6379/0

BLOB_BACKEND=local
BLOB_LOCAL_DIR=${HOME}/.cache/researchflow/blobs

JWT_SECRET=${JWT_VAL}
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=10080

OPENAI_API_KEY=${OPENAI_KEY:-}
ANTHROPIC_API_KEY=${ANTHROPIC_KEY:-}
GOOGLE_API_KEY=${GOOGLE_KEY:-}

DEFAULT_CHEAP_MODEL=gpt-4o-mini
DEFAULT_QUALITY_MODEL=gpt-5.4-mini
DEFAULT_REASONING_MODEL=gpt-5.4
DEFAULT_LLM_PROVIDER=openai

DEFAULT_EMBEDDING_PROVIDER=${EMB_PROVIDER}
DEFAULT_EMBEDDING_MODEL=${EMB_MODEL}
DEFAULT_EMBEDDING_DIM=${EMB_DIM}

IMAGE_GEN_PROVIDER=openai
PDF_PARSER=marker
INGESTION_MODE=rss

RESEND_API_KEY=${RESEND_KEY:-}
EMAIL_FROM=noreply@researchflow.ai
EMAIL_FROM_NAME=ResearchFlow

LANGSMITH_API_KEY=${LANGSMITH_KEY:-}
LANGSMITH_PROJECT=researchflow
LANGCHAIN_TRACING_V2=false

BREAKTHROUGH_THRESHOLD=0.88

# arXiv RSS updates at midnight ET; new papers land Tue–Fri (after Mon–Thu announcements).
# Run ingestion at 05:00 UTC (01:00 ET) on Tue–Fri only (days 2–5).
# Weekly maintenance jobs run on Sunday 05:00/05:30 UTC (no arXiv activity that day).
INGESTION_CRON=0 5 * * 2-5
CLUSTERING_CRON=0 5 * * 0
CROSS_NAMESPACE_CRON=30 5 * * 0

CORS_ORIGINS=["http://localhost:3000"]
ENVIRONMENT=local
DEBUG=true
LOG_LEVEL=DEBUG
ENVEOF

mv "$TMP_ENV" "$ENV_FILE"
success ".env.local written"

echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > "$SCRIPT_DIR/frontend/.env.local"
success "frontend/.env.local written"

# ── [4/7] Start database ──────────────────────────────────────────────────────
step "[4/7] Starting database (PostgreSQL + pgvector)"
detect_dc
start_db

# ── [5/7] Python venv + deps ──────────────────────────────────────────────────
step "[5/7] Setting up Python backend"

# Detect existing venv Python version
VENV_PYTHON_VER=""
[[ -f "$VENV_DIR/bin/python" ]] && \
    VENV_PYTHON_VER=$("$VENV_DIR/bin/python" -c \
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)

TARGET_PYTHON_VER=$("$PYTHON_CMD" -c \
    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

if [[ -d "$VENV_DIR" && "$VENV_PYTHON_VER" == "$TARGET_PYTHON_VER" ]]; then
    success "Virtualenv (Python $VENV_PYTHON_VER) already exists — reusing"
else
    if [[ -d "$VENV_DIR" && -n "$VENV_PYTHON_VER" && "$VENV_PYTHON_VER" != "$TARGET_PYTHON_VER" ]]; then
        warn "Existing venv is Python $VENV_PYTHON_VER — rebuilding for $TARGET_PYTHON_VER"
        rm -rf "$VENV_DIR"
    fi
    info "Creating virtualenv (Python $TARGET_PYTHON_VER) at backend/.venv …"
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    success "Virtualenv created (Python $TARGET_PYTHON_VER)"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip
info "Installing backend dependencies (first run: 2–3 min) …"
if ! pip install -r "$SCRIPT_DIR/backend/requirements.txt"; then
    error "pip install failed — see output above. Fix the error and re-run."
    exit 1
fi
success "Backend dependencies installed"

# ── [6/7] Seed database ───────────────────────────────────────────────────────
step "[6/7] Seeding database"

info "Running seed script (idempotent — safe to re-run) …"
echo ""
cd "$SCRIPT_DIR/backend"
PYTHONPATH="$SCRIPT_DIR/backend" python scripts/seed_db.py
cd "$SCRIPT_DIR"
echo ""

# ── [7/7] Frontend deps ───────────────────────────────────────────────────────
step "[7/7] Installing frontend dependencies"

cd "$SCRIPT_DIR/frontend"
npm install --loglevel=error 2>&1 || {
    error "npm install failed. See output above."
    echo -e "  ${DIM}Fix: rm -rf frontend/node_modules && ./setup.sh --setup${RESET}"
    exit 1
}
success "Frontend dependencies installed"
cd "$SCRIPT_DIR"

# ── Write state file ──────────────────────────────────────────────────────────
write_state

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN}║   ✓  Setup complete!                                 ║${RESET}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${BOLD}Test user:${RESET}  test@researchflow.ai  /  ResearchFlow2024!"
echo ""
echo -e "  ${BOLD}API keys:${RESET}"
is_real_key "$OPENAI_KEY"    && echo -e "    ${GREEN}✓${RESET}  OpenAI"    || echo -e "    ${YELLOW}–${RESET}  OpenAI     (not set)"
is_real_key "$GOOGLE_KEY"    && echo -e "    ${GREEN}✓${RESET}  Google AI" || echo -e "    ${YELLOW}–${RESET}  Google AI  (not set)"
is_real_key "$ANTHROPIC_KEY" && echo -e "    ${GREEN}✓${RESET}  Anthropic" || echo -e "    ${DIM}–  Anthropic (optional)${RESET}"
is_real_key "$RESEND_KEY"    && echo -e "    ${GREEN}✓${RESET}  Resend"    || echo -e "    ${DIM}–  Resend    (optional)${RESET}"
is_real_key "$LANGSMITH_KEY" && echo -e "    ${GREEN}✓${RESET}  LangSmith" || echo -e "    ${DIM}–  LangSmith (optional)${RESET}"
echo ""
echo -e "  ${DIM}Next time just run:  ${RESET}${BOLD}./setup.sh${RESET}${DIM}  — it will skip setup and start directly.${RESET}"
echo ""

read -rp "  Launch backend + frontend now? [Y/n] " launch
echo ""
if [[ ! "${launch,,}" =~ ^n ]]; then
    launch_servers "$VENV_DIR"
else
    echo -e "  To start later, run: ${BOLD}./setup.sh${RESET}  ${DIM}(or ./setup.sh --run)${RESET}"
fi
