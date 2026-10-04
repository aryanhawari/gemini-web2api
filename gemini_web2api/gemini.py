"""Core Gemini StreamGenerate protocol.

Covers: endpoint URL building (build label + account prefix), the sparse f.req
payload, browser-style headers with optional cookie/SAPISIDHASH auth, the weird
")]}'"-prefixed multi-line JSON response parsing, progressive-text streaming,
build-label (405) and xsrf (400) recovery, and bounded retries with backoff.
"""

import hashlib
import json
import random
import re
import threading
import time
import urllib.request
import uuid
from urllib.parse import urlencode

from .config import CookieStore

GEMINI_BASE = "https://gemini.google.com"
STREAM_PATH = "/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
UPLOAD_BASE = "https://content-push.googleapis.com/upload/"

BL_RE = re.compile(r"boq_[A-Za-z0-9-]+_\d{8}\.\d+_p\d+")
CFB2H_RE = re.compile(r'"cfb2h":"([^"]+)"')
SNLM0E_RE = re.compile(r'"SNlM0e":"([^"]+)"')
QKIAYE_RE = re.compile(r'"qKIAYe":"([^"]+)"')
YLRO7B_RE = re.compile(r'"Ylro7b":"([^"]+)"')

PREFIX_MARK = ")]}'"
MIN_FRAME_LINE_LEN = 200  # candidate frames live on long lines; shorter ones are noise

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

try:
    import httpx  # type: ignore
    HAVE_HTTPX = True
except ImportError:  # pragma: no cover - httpx optional (urllib fallback)
    HAVE_HTTPX = False

try:
    from curl_cffi.requests import Session as CurlSession  # type: ignore
    HAVE_CURL_CFFI = True
except ImportError:  # pragma: no cover - curl_cffi optional but needed for images
    HAVE_CURL_CFFI = False

try:
    import h2  # noqa: F401  # type: ignore
    HAVE_H2 = True
except ImportError:  # pragma: no cover - h2 optional; HTTP/2 only when present
    HAVE_H2 = False

KEEPALIVE_EXPIRY_SEC = 600  # hold pooled TLS connections far beyond httpx's 5s default


class GeminiError(RuntimeError):
    """Fatal upstream/protocol error."""


class RetryableError(GeminiError):
    """Error worth retrying after refreshing some state."""

    def __init__(self, message, kind="generic"):
        super().__init__(message)
        self.kind = kind  # "bl" | "xsrf" | "empty" | "rewrite" | "generic"


class RateLimitedError(GeminiError):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def _header(headers, name):
    """Case-insensitive header lookup across plain dicts and httpx.Headers."""
    if not headers:
        return None
    lname = name.lower()
    for key, value in headers.items():
        if key.lower() == lname:
            return value
    return None


# --------------------------------------------------------------------------
# Payload construction
# --------------------------------------------------------------------------

def build_inner(prompt, file_refs=None, model_info=None, temporary_chats=False, lang="en"):
    """Builds the sparse inner array carried as a JSON string inside f.req.

    file_refs items are either plain refs or [ref, filename] pairs; the current
    upstream format expects each as [[ref], filename].
    """
    entries = []
    for item in (file_refs or []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            ref, fname = item[0], item[1]
        else:
            ref, fname = item, None
        if not fname:
            fname = f"input_{random.randint(1000000, 9999999)}.png"
        entries.append([[ref], fname])
    inner = [None] * 102
    inner[0] = [prompt, 0, None, entries or None, None, None, 0]  # user message + file refs
    inner[1] = [lang]
    inner[2] = ["", "", "", None, None, None, None, None, None, ""]  # conversation context
    inner[6] = [0]
    inner[7] = 1   # streaming enable
    inner[10] = 1
    inner[11] = 0  # safety filter level
    inner[17] = [[model_info.think if model_info else 4]]  # thinking depth
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    inner[41] = [1] if temporary_chats else [2]  # history persistence
    if temporary_chats:
        inner[45] = 1
    inner[53] = 0
    inner[59] = str(uuid.uuid4())  # request id
    inner[61] = []
    inner[68] = 1
    inner[79] = model_info.mode if model_info else 1  # MODEL_CATEGORY -> model selection
    if model_info and model_info.extra:
        for idx, value in model_info.extra.items():
            inner[idx] = value
    return inner


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------

_CODE_REF_RE = re.compile(r"```([A-Za-z0-9_+#-]*)\?code_[a-z_]+&code_event_index=\d+")
_CARD_CONTENT_RE = re.compile(r"https?://googleusercontent\.com/card_content/\d+/?\S*")


def clean_text(text):
    """Strips code-execution artifacts and card_content links Gemini injects."""
    if not text:
        return text
    text = _CODE_REF_RE.sub(r"```\1", text)
    text = _CARD_CONTENT_RE.sub("", text)
    return text


def _safe_json(raw):
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def _iter_frames(data):
    """Yields candidate frames (lists tagged with a string first element)."""
    if not isinstance(data, list) or not data:
        return
    if isinstance(data[0], str):
        yield data
        return
    for item in data:
        yield from _iter_frames(item)


def _check_error_frames(data):
    for frame in _iter_frames(data):
        if isinstance(frame, list) and frame and frame[0] == "er":
            raise GeminiError("Gemini upstream returned an error frame.")


def _candidate_text(candidate):
    """Extracts text from one candidate.

    Shapes seen upstream:
      - candidate[1] is a plain string (older snapshots)
      - candidate[1] is a list of strings  (current: ["OK-", "TEST"])
      - candidate[1] is a list of parts where part[1] is a string or a
        list of strings (legacy nested format)
    """
    if not isinstance(candidate, list) or len(candidate) < 2:
        return ""
    payload = candidate[1]
    if isinstance(payload, str):
        return payload
    texts = []
    if isinstance(payload, list):
        for part in payload:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, list) and len(part) > 1:
                t = part[1]
                if isinstance(t, str):
                    texts.append(t)
                elif isinstance(t, list):
                    texts.extend(s for s in t if isinstance(s, str))
    return "".join(texts)


def _thought_text(candidate):
    """Extracts the thinking-trace text from one candidate ('' if none).

    Thoughts live at candidate[37] (2026 shape): the markdown snapshot is the
    [37][0] subtree — its longest string is the thought; the whole subtree is
    thought text. Code-execution/tool chips park their UI strings (icon
    URLs, executed code, captions) under [37][1], which must be ignored —
    those otherwise leak in as fake reasoning. A 50-char floor drops trivial
    leftovers; real thought snapshots are full markdown texts.
    """
    if not isinstance(candidate, list) or len(candidate) <= 37:
        return ""
    root = candidate[37]
    if not (isinstance(root, list) and root and isinstance(root[0], list)):
        return ""
    best = ""
    stack = [root[0]]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            if "://" not in node and len(node) > len(best) and len(node) >= 50:
                best = node
        elif isinstance(node, list):
            stack.extend(node)
    return best


def extract_line_parts(raw_line, min_line_len=0):
    """Returns (answer_text, thought_text) found in one response line.

    Gemini sends progressively longer full texts per event, so per line the
    longest candidate is the current answer; thoughts behave the same way.
    """
    line = raw_line.strip()
    if not line or line == PREFIX_MARK:
        return "", ""
    if line.startswith(PREFIX_MARK):
        line = line[len(PREFIX_MARK):].lstrip()
    try:
        data = json.loads(line)
    except ValueError:
        return "", ""
    _check_error_frames(data)
    if min_line_len and len(line) < min_line_len:
        return "", ""
    best_text = ""
    best_thought = ""
    for frame in _iter_frames(data):
        if not (isinstance(frame, list) and len(frame) >= 3 and frame[0] == "wrb.fr"):
            continue
        if not isinstance(frame[2], str):
            continue
        inner = _safe_json(frame[2])
        if not (isinstance(inner, list) and len(inner) > 4 and isinstance(inner[4], list)):
            continue
        for candidate in inner[4] or []:
            text = _candidate_text(candidate)
            if len(text) > len(best_text):
                best_text = text
            thought = _thought_text(candidate)
            if len(thought) > len(best_thought):
                best_thought = thought
    return best_text, best_thought


def extract_line_text(raw_line, min_line_len=0):
    """Returns the longest candidate text found in one response line ('' if none)."""
    return extract_line_parts(raw_line, min_line_len)[0]


def parse_response(body):
    """Parses a complete (non-streamed) StreamGenerate body into the final text."""
    if "BardErrorInfo" in body:
        raise GeminiError("Gemini upstream rejected the request (BardErrorInfo).")
    best = ""
    for raw in body.splitlines():
        text = extract_line_text(raw, min_line_len=MIN_FRAME_LINE_LEN)
        if len(text) > len(best):
            best = text
    return best


def iter_events(lines):
    """Converts raw response lines into ('thought'|'text', delta) events.

    Each event carries the full text so far; a delta is the new suffix. A
    non-prefix rewrite of the answer is a protocol error worth retrying from
    scratch; a rewrite of the thinking trace (new thought section) is emitted
    as a fresh paragraph instead.
    """
    prev_text = ""
    prev_thought = ""
    for raw in lines:
        text, thought = extract_line_parts(raw)
        if thought:
            if thought.startswith(prev_thought):
                delta = thought[len(prev_thought):]
            elif len(thought) > len(prev_thought):
                delta = "\n\n" + thought
            else:
                delta = ""
            prev_thought = thought
            if delta:
                yield ("thought", delta)
        text = clean_text(text)
        if not text:
            continue
        if not text.startswith(prev_text):
            raise RetryableError("Gemini rewrote its earlier output mid-stream.", kind="rewrite")
        if len(text) > len(prev_text):
            yield ("text", text[len(prev_text):])
            prev_text = text


def iter_deltas(lines):
    """Text-only view of iter_events (answer content deltas)."""
    for kind, delta in iter_events(lines):
        if kind == "text":
            yield delta


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

def extract_build_label(html):
    """Build label = WIZ_global_data.cfb2h (e.g. boq_gemini-web-uiserver_...),
    with a raw boq_* scan of the HTML as fallback for older page layouts."""
    match = CFB2H_RE.search(html)
    if match:
        return match.group(1)
    match = BL_RE.search(html)
    return match.group(0) if match else None


class GeminiClient:
    def __init__(self, config):
        self.cfg = config
        if config.cookie_file:
            self.cookies = CookieStore(config.cookie_file)
        elif getattr(config, "cookie_string", None):
            from .config import StaticCookieStore
            self.cookies = StaticCookieStore(config.cookie_string)
        else:
            self.cookies = None
        self._http_client = None
        self._curl_client = None
        self._bl_lock = threading.Lock()
        self._xsrf_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._xsrf = config.xsrf_token

    # -- low-level HTTP ----------------------------------------------------

    def _curl(self):
        """curl_cffi session with Chrome TLS impersonation.

        Google's bot detection treats file-attached requests more strictly;
        plain httpx TLS fingerprints get soft-rejected (BardErrorInfo) there,
        while an impersonated Chrome fingerprint passes. None if not installed.
        """
        if self._curl_client is None and HAVE_CURL_CFFI:
            kwargs = {"timeout": self.cfg.request_timeout_sec}
            if self.cfg.proxy:
                kwargs["proxies"] = {"http": self.cfg.proxy, "https": self.cfg.proxy}
            self._curl_client = CurlSession(impersonate="chrome", **kwargs)
        return self._curl_client

    def _use_curl_for_files(self, file_refs):
        return bool(file_refs) and self._curl() is not None

    def _http(self):
        """Shared httpx client (None when httpx is not installed).

        HTTP/2 is enabled when the h2 package is available (fewer handshakes
        under concurrency); keep-alive expiry is raised from httpx's 5s
        default so an idle minute between chats does not cost a fresh
        TCP+TLS handshake.
        """
        if self._http_client is None and HAVE_HTTPX:
            common = dict(
                timeout=self.cfg.request_timeout_sec,
                follow_redirects=True,
                http2=HAVE_H2,
                limits=httpx.Limits(keepalive_expiry=KEEPALIVE_EXPIRY_SEC),
                headers={"User-Agent": DEFAULT_USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
            )
            if self.cfg.proxy:
                try:
                    self._http_client = httpx.Client(proxy=self.cfg.proxy, **common)
                except TypeError:  # httpx < 0.26
                    self._http_client = httpx.Client(proxies=self.cfg.proxy, **common)
            else:
                self._http_client = httpx.Client(**common)
        return self._http_client

    def _urllib_request(self, method, url, body=None, headers=None):
        """Non-streaming fallback used when httpx is unavailable."""
        req = urllib.request.Request(url, data=body, method=method)
        merged = {"User-Agent": DEFAULT_USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
        merged.update(headers or {})
        for key, value in merged.items():
            req.add_header(key, value)
        handlers = []
        if self.cfg.proxy:
            handlers.append(urllib.request.ProxyHandler(
                {"http": self.cfg.proxy, "https": self.cfg.proxy}))
        opener = urllib.request.build_opener(*handlers)
        with opener.open(req, timeout=self.cfg.request_timeout_sec) as resp:
            return resp.read().decode("utf-8", "replace")

    def _prefix(self):
        return f"/u/{self.cfg.auth_user}" if self.cfg.auth_user is not None else ""

    def _common_headers(self):
        headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
        cookie, sapisid = self.cookies.get() if self.cookies else ("", "")
        if cookie:
            headers["Cookie"] = cookie
        if sapisid:
            ts = int(time.time())
            digest = hashlib.sha1(f"{ts} {sapisid} {GEMINI_BASE}".encode()).hexdigest()
            headers["Authorization"] = f"SAPISIDHASH {ts}_{digest}"
        return headers

    # -- page-derived state (build label, xsrf token) -----------------------

    def fetch_page(self, path="/app"):
        url = f"{GEMINI_BASE}{self._prefix()}{path}"
        http = self._http()
        if http is not None:
            resp = http.get(url, headers=self._common_headers())
            resp.raise_for_status()
            return resp.text
        return self._urllib_request("GET", url)

    def refresh_state(self):
        """One /app fetch refreshes the build label AND the xsrf token.

        Both are embedded in the same page, so a combined fetch halves the
        warmup/refresh cost versus a bl-only fetch followed by an xsrf fetch.
        Raises GeminiError when no build label is found (page layout change).
        """
        with self._state_lock:
            html = self.fetch_page("/app")
            label = extract_build_label(html)
            if not label:
                raise GeminiError(
                    "Could not extract build label (cfb2h / boq_*) from "
                    "gemini.google.com/app — page layout may have changed.")
            self.cfg.gemini_bl = label
            if self.cookies and self.cookies.get()[0]:
                match = SNLM0E_RE.search(html)
                if match:
                    with self._xsrf_lock:
                        self._xsrf = match.group(1)
            return label

    def refresh_bl(self):
        self.cfg.gemini_bl = self.refresh_state()
        return self.cfg.gemini_bl

    def ensure_bl(self):
        if self.cfg.gemini_bl:
            return self.cfg.gemini_bl
        with self._bl_lock:
            if not self.cfg.gemini_bl:
                self.refresh_state()
            return self.cfg.gemini_bl

    def get_xsrf(self, force=False):
        """SNlM0e token; only fetchable (and only needed) with cookies."""
        if self._xsrf and not force:
            return self._xsrf
        with self._xsrf_lock:
            if self._xsrf and not force:
                return self._xsrf
            if not (self.cookies and self.cookies.get()[0]):
                return None
            match = SNLM0E_RE.search(self.fetch_page("/app"))
            self._xsrf = match.group(1) if match else None
            return self._xsrf

    # -- request building ---------------------------------------------------

    def build_url(self):
        bl = self.ensure_bl()
        reqid = int(time.time()) % 1000000
        return (f"{GEMINI_BASE}{self._prefix()}{STREAM_PATH}"
                f"?bl={bl}&hl=en&_reqid={reqid}&rt=c")

    def build_body(self, inner):
        freq = json.dumps(
            [None, json.dumps(inner, separators=(",", ":"), ensure_ascii=False)],
            separators=(",", ":"), ensure_ascii=False)
        fields = {"f.req": freq}
        xsrf = self._xsrf
        if xsrf:
            fields["at"] = xsrf
        return urlencode(fields).encode("utf-8")

    def _post_headers(self):
        headers = self._common_headers()
        headers.update({
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": GEMINI_BASE,
            "Referer": f"{GEMINI_BASE}{self._prefix()}/app",
            "X-Same-Domain": "1",
        })
        if self.cfg.auth_user is not None:
            headers["X-Goog-AuthUser"] = str(self.cfg.auth_user)
        return headers

    @staticmethod
    def _raise_for_status(status, snippet="", headers=None):
        if status == 200:
            return
        if status == 405:
            raise RetryableError("upstream 405 — build label expired", kind="bl")
        if status == 400:
            raise RetryableError("upstream 400 — missing/invalid xsrf token", kind="xsrf")
        if status == 403:
            raise GeminiError(f"upstream 403 — cookies missing or rejected. {snippet}")
        if status == 429:
            raw = _header(headers, "retry-after")
            retry_after = None
            if raw:
                try:
                    retry_after = float(raw)
                except ValueError:
                    retry_after = None
            raise RateLimitedError("upstream 429 — rate limited", retry_after)
        raise GeminiError(f"upstream HTTP {status}: {snippet}")

    def _post_once_curl(self, inner):
        """Same request via the curl_cffi session (Chrome TLS fingerprint)."""
        url = self.build_url()
        body = self.build_body(inner)
        resp = self._curl().post(url, data=body, headers=self._post_headers())
        self._raise_for_status(resp.status_code, resp.text[:300],
                               dict(resp.headers.items()))
        return resp.text

    def _prepare_xsrf(self):
        """Proactively fetch the xsrf token when cookies exist (saves a 400 round-trip)."""
        if self.cookies and not self._xsrf:
            try:
                self.get_xsrf()
            except GeminiError:
                pass

    # -- execution ----------------------------------------------------------

    def _post_once(self, inner):
        url = self.build_url()
        body = self.build_body(inner)
        headers = self._post_headers()
        http = self._http()
        if http is not None:
            resp = http.post(url, content=body, headers=headers)
            self._raise_for_status(resp.status_code, resp.text[:300], resp.headers)
            return resp.text
        try:
            return self._urllib_request("POST", url, body=body, headers=headers)
        except urllib.error.HTTPError as exc:  # pragma: no cover
            snippet = exc.read().decode("utf-8", "replace")[:300]
            self._raise_for_status(exc.code, snippet, dict(exc.headers))
            raise

    def _post_stream_once(self, inner):
        """Generator of raw response lines (requires httpx)."""
        url = self.build_url()
        body = self.build_body(inner)
        headers = self._post_headers()
        with self._http().stream("POST", url, content=body, headers=headers) as resp:
            self._raise_for_status(resp.status_code, "", resp.headers)
            for raw in resp.iter_lines():
                if raw:
                    yield raw

    def _backoff(self, attempt, exc):
        delay = self.cfg.retry_delay_sec * (2 ** attempt)
        if isinstance(exc, RateLimitedError) and exc.retry_after:
            delay = max(delay, exc.retry_after)
        return delay + random.uniform(0, 0.5)

    def _recover(self, exc):
        if isinstance(exc, RetryableError) and exc.kind == "bl":
            try:
                self.refresh_bl()
            except GeminiError:
                pass
        elif isinstance(exc, RetryableError) and exc.kind == "xsrf":
            try:
                self.get_xsrf(force=True)
            except GeminiError:
                pass

    def generate(self, prompt, file_refs=None, model_info=None):
        """Blocking generation with bounded retries. Returns the final text."""
        self._prepare_xsrf()
        use_curl = self._use_curl_for_files(file_refs)
        inner = build_inner(prompt, file_refs or [], model_info,
                            temporary_chats=self.cfg.temporary_chats)
        last = None
        attempts = max(1, self.cfg.retry_attempts)
        for attempt in range(attempts):
            try:
                if use_curl:
                    body = self._post_once_curl(inner)
                else:
                    body = self._post_once(inner)
                text = clean_text(parse_response(body))
                if not text:
                    raise RetryableError("Gemini returned an empty response.", kind="empty")
                return text
            except (RetryableError, RateLimitedError, GeminiError) as exc:
                last = exc
                self._recover(exc)
                if attempt < attempts - 1:
                    time.sleep(self._backoff(attempt, last))
        raise last

    def stream_events(self, prompt, file_refs=None, model_info=None):
        """Yields ('thought', delta) / ('text', delta) events.

        Retries only before the first text delta; thought-only progress does
        not block a retry (a repeated thinking trace is cosmetic, an empty
        answer is not).
        """
        if self._http() is None:  # no httpx -> non-streaming fallback
            yield ("text", self.generate(prompt, file_refs, model_info))
            return
        if self._use_curl_for_files(file_refs):
            # curl_cffi transport is used non-streamed for file requests;
            # the full answer arrives as a single delta.
            yield ("text", self.generate(prompt, file_refs, model_info))
            return
        self._prepare_xsrf()
        inner = build_inner(prompt, file_refs or [], model_info,
                            temporary_chats=self.cfg.temporary_chats)
        last = None
        attempts = max(1, self.cfg.retry_attempts)
        for attempt in range(attempts):
            text_emitted = False
            try:
                for kind, delta in iter_events(self._post_stream_once(inner)):
                    if kind == "text":
                        text_emitted = True
                    yield kind, delta
                if not text_emitted:
                    raise RetryableError("Gemini returned an empty stream.", kind="empty")
                return
            except (RetryableError, RateLimitedError, GeminiError) as exc:
                last = exc
                if text_emitted:
                    raise
                self._recover(exc)
                if attempt < attempts - 1:
                    time.sleep(self._backoff(attempt, last))
        raise last

    def stream_generate(self, prompt, file_refs=None, model_info=None):
        """Yields answer content deltas (thinking trace filtered out)."""
        for kind, delta in self.stream_events(prompt, file_refs, model_info):
            if kind == "text":
                yield delta
