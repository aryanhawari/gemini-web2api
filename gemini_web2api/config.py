"""Configuration loading, cookie-file parsing and CLI argument handling.

Config search order: --config flag -> GEMINI_WEB2API_CONFIG env -> ./config.json
-> ~/.config/gemini-web2api/config.json. Missing files are fine (defaults win).
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, fields
from typing import List, Optional

ENV_CONFIG_VAR = "GEMINI_WEB2API_CONFIG"
LOCAL_CONFIG = "config.json"
USER_CONFIG = os.path.join("~", ".config", "gemini-web2api", "config.json")

INT_FIELDS = {"port", "retry_attempts", "request_timeout_sec", "auth_user",
              "state_refresh_sec", "pool_workers"}
FLOAT_FIELDS = {"retry_delay_sec", "queue_wait_sec"}
BOOL_FIELDS = {"log_requests", "temporary_chats"}


@dataclass
class Config:
    port: int = 8000
    host: str = "0.0.0.0"
    retry_attempts: int = 3
    retry_delay_sec: float = 2.0
    request_timeout_sec: int = 180
    gemini_bl: Optional[str] = None          # auto-refreshed at runtime
    auth_user: Optional[int] = None          # /u/N/ account index
    xsrf_token: Optional[str] = None         # SNlM0e
    default_model: str = "gemini-3.6-flash"
    api_keys: List[str] = None               # empty list = no auth
    cookie_file: Optional[str] = None        # cookie.txt path
    cookie_string: Optional[str] = None      # full Cookie header (env COOKIE_STRING)
    proxy: Optional[str] = None              # e.g. http://127.0.0.1:7890
    log_requests: bool = True
    temporary_chats: bool = False            # true = history not saved on the account
    state_refresh_sec: int = 300             # background bl/xsrf refresh cadence (0 = off)
    max_concurrent_requests: int = 8         # upstream concurrency cap
    queue_wait_sec: float = 30.0             # max wait for a free slot before 429 (0 = forever)
    pool_workers: int = 64                   # HTTP worker threads serving connections
    max_body_mb: float = 10.0                # request body size limit
    strict_models: bool = True               # unknown model name -> 400 (else fallback)

    def __post_init__(self):
        if self.api_keys is None:
            self.api_keys = []


_CONFIG_FIELDS = {f.name for f in fields(Config)}

ENV_LIST_FIELDS = {"api_keys"}


def load_env_file(path=".env"):
    """Parses a .env file (KEY=VALUE lines, # comments, optional quotes). -> dict."""
    env = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    env[key] = value
    except OSError:
        pass
    return env


def _apply_env_overrides(cfg: Config, env: dict) -> None:
    """Applies PROXY_API_KEY / PROXY_API_KEYS / tuning env vars (env > config.json)."""
    keys = list(cfg.api_keys)
    raw_single = env.get("PROXY_API_KEY", "").strip()
    if raw_single and raw_single not in keys:
        keys.append(raw_single)
    raw_multi = env.get("PROXY_API_KEYS", "").strip()
    if raw_multi:
        for k in raw_multi.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    cfg.api_keys = keys

    raw_cookie = env.get("COOKIE_STRING", "").strip()
    if raw_cookie:
        cfg.cookie_string = raw_cookie
        if not cfg.cookie_file:
            cfg.cookie_file = None  # env cookie wins when no file configured

    if env.get("RETRY_ATTEMPTS", "").strip().isdigit():
        cfg.retry_attempts = int(env["RETRY_ATTEMPTS"])
    if env.get("STATE_REFRESH_SEC", "").strip().isdigit():
        cfg.state_refresh_sec = int(env["STATE_REFRESH_SEC"])
    if env.get("REQUEST_TIMEOUT_SEC", "").strip().isdigit():
        cfg.request_timeout_sec = int(env["REQUEST_TIMEOUT_SEC"])
    if env.get("MAX_CONCURRENT_REQUESTS", "").strip().isdigit():
        cfg.max_concurrent_requests = int(env["MAX_CONCURRENT_REQUESTS"])
    if env.get("POOL_WORKERS", "").strip().isdigit():
        cfg.pool_workers = int(env["POOL_WORKERS"])
    if env.get("PORT", "").strip().isdigit():
        cfg.port = int(env["PORT"])
    for name in ("RETRY_DELAY_SEC", "MAX_BODY_MB", "QUEUE_WAIT_SEC"):
        raw = env.get(name, "").strip()
        if raw:
            try:
                setattr(cfg, name.lower(), float(raw))
            except ValueError:
                pass


def _apply(cfg: Config, data: dict) -> None:
    for key, value in (data or {}).items():
        if key not in _CONFIG_FIELDS or value is None:
            continue
        if key == "api_keys":
            cfg.api_keys = [str(k) for k in value] if isinstance(value, list) else [str(value)]
        elif key in INT_FIELDS:
            try:
                setattr(cfg, key, int(value))
            except (TypeError, ValueError):
                pass
        elif key in FLOAT_FIELDS:
            try:
                setattr(cfg, key, float(value))
            except (TypeError, ValueError):
                pass
        elif key in BOOL_FIELDS:
            setattr(cfg, key, bool(value))
        else:
            setattr(cfg, key, str(value))


def find_config(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    env = os.environ.get(ENV_CONFIG_VAR)
    if env:
        return env
    for cand in (LOCAL_CONFIG, os.path.expanduser(USER_CONFIG)):
        if os.path.isfile(cand):
            return cand
    return None


def load_config(explicit: Optional[str] = None) -> Config:
    cfg = Config()
    path = find_config(explicit)
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            _apply(cfg, json.load(fh))
        cfg.config_path = path
    env = dict(os.environ)
    env.update(load_env_file(".env"))
    _apply_env_overrides(cfg, env)
    return cfg


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="gemini-web2api",
        description="OpenAI-compatible API server for the Gemini web app (reverse-engineered).",
    )
    parser.add_argument("--port", type=int, help="listen port (default 8000)")
    parser.add_argument("--host", help="bind address (default 0.0.0.0)")
    parser.add_argument("--config", help="path to config.json")
    parser.add_argument("--cookie-file", help="path to cookie file (header string or JSON)")
    parser.add_argument("--proxy", help="proxy URL, e.g. http://127.0.0.1:7890")
    return parser.parse_args(argv)


def apply_args(cfg: Config, args) -> Config:
    if getattr(args, "port", None) is not None:
        cfg.port = args.port
    if getattr(args, "host", None):
        cfg.host = args.host
    if getattr(args, "cookie_file", None):
        cfg.cookie_file = args.cookie_file
    if getattr(args, "proxy", None):
        cfg.proxy = args.proxy
    return cfg


class StaticCookieStore:
    """Cookie source for a fixed Cookie header string (e.g. env COOKIE_STRING)."""

    def __init__(self, cookie: str):
        self.cookie = cookie
        m = re.search(r"SAPISID=([^;\s]+)", cookie)
        self.sapisid = m.group(1).strip() if m else ""

    def get(self):
        return self.cookie, self.sapisid


class CookieStore:
    """Reads a cookie file (plain Cookie-header string or JSON {cookie, sapisid}).

    The file is re-read whenever its mtime changes, so swapping cookies does not
    require a server restart.
    """

    SAPISID_RE = re.compile(r"SAPISID=([^;\s]+)")

    def __init__(self, path: str):
        self.path = path
        self._mtime = None
        self._cookie = ""
        self._sapisid = ""

    def get(self):
        """Returns (cookie_header, sapisid); either may be empty."""
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return "", ""
        if self._mtime is None or mtime != self._mtime:
            try:
                self._load()
            except OSError:
                return "", ""
            self._mtime = mtime
        return self._cookie, self._sapisid

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as fh:
            raw = fh.read().strip()
        cookie, sapisid = raw, ""
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                cookie = str(data.get("cookie", "")).strip()
                sapisid = str(data.get("sapisid", "")).strip()
        except ValueError:
            pass
        if cookie and not sapisid:
            m = self.SAPISID_RE.search(cookie)
            if m:
                sapisid = m.group(1).strip()
        self._cookie, self._sapisid = cookie, sapisid
