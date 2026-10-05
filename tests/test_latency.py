"""Latency / high-traffic regression tests.

All upstream work is faked; these tests exercise the admission gate, the
tail-latency hedge, SSE keep-alives and the fast-fail overload paths that keep
client-visible latency bounded when traffic spikes.
"""

import json
import os
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_web2api.config import Config
from gemini_web2api.gemini import GeminiClient, RetryableError, UpstreamTimeoutError
from gemini_web2api.limiter import AdaptiveConcurrency
from gemini_web2api.server import App, make_handler


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------

class SlowFakeGemini:
    """Streams a fixed text with a per-chunk delay; tracks concurrency."""

    def __init__(self, text="hello world", delay=0.02):
        self.text = text
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.peak_active = 0
        self._lock = threading.Lock()

    def generate(self, prompt, file_refs=None, model_info=None):
        self.calls += 1
        return self.text

    def stream_events(self, prompt, file_refs=None, model_info=None):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            for i in range(0, len(self.text), 4):
                time.sleep(self.delay)
                yield ("text", self.text[i:i + 4])
        finally:
            with self._lock:
                self.active -= 1

    def stream_generate(self, prompt, file_refs=None, model_info=None):
        for kind, delta in self.stream_events(prompt, file_refs, model_info):
            if kind == "text":
                yield delta


class SlowFirstByteGemini(SlowFakeGemini):
    """Silent for a while, then streams (exercise SSE keep-alives)."""

    def __init__(self, text="delayed answer", delay=0.4):
        super().__init__(text=text, delay=0.01)
        self.first_delay = delay

    def stream_events(self, prompt, file_refs=None, model_info=None):
        time.sleep(self.first_delay)
        for event in super().stream_events(prompt, file_refs, model_info):
            yield event


class BoomGemini:
    def generate(self, prompt, file_refs=None, model_info=None):
        raise UpstreamTimeoutError("upstream stalled")

    def stream_events(self, prompt, file_refs=None, model_info=None):
        raise UpstreamTimeoutError("upstream stalled")
        yield  # pragma: no cover


class NoopUploader:
    def upload(self, data, mime):
        return "/uploaded/ref"


def make_app(fake=None, **overrides):
    config = Config(**overrides)
    return App(config, gemini=fake or SlowFakeGemini(), uploader=NoopUploader(),
               image_engine=False)


def make_server(app):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def request(port, method, path, body=None, timeout=20):
    conn = HTTPConnection("127.0.0.1", port, timeout=timeout)
    payload = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, payload, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    headers = dict(resp.getheaders())
    status = resp.status
    conn.close()
    return status, data, headers


CHAT = {"model": "gemini-3.6-flash",
        "messages": [{"role": "user", "content": "hi"}]}


# ---------------------------------------------------------------------------
# Adaptive admission gate
# ---------------------------------------------------------------------------

class AdaptiveConcurrencyTest(unittest.TestCase):

    def test_grows_when_clients_wait_and_shrinks_on_rate_limit(self):
        gate = AdaptiveConcurrency(initial=2, minimum=1, maximum=16)
        self.assertTrue(gate.acquire(timeout=1))
        self.assertTrue(gate.acquire(timeout=1))
        waiter_got = threading.Event()
        waiter_done = threading.Event()

        def waiter():
            if gate.acquire(timeout=5):
                waiter_got.set()
                gate.release(neutral=True)
            waiter_done.set()

        thread = threading.Thread(target=waiter)
        thread.start()
        try:
            deadline = time.monotonic() + 2
            while gate.waiting == 0 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertGreaterEqual(gate.waiting, 1)
            gate.release(success=True)  # success with a backlog -> grow
            self.assertTrue(waiter_got.wait(2))
            self.assertTrue(waiter_done.wait(2))
        finally:
            thread.join(timeout=2)
        self.assertGreater(gate.limit, 2)
        grown = gate.limit
        # the waiter's neutral release must not have touched the limit
        self.assertGreater(gate.increases, 0)
        self.assertTrue(gate.acquire(timeout=0.5))
        gate.release(success=False, pressure="rate_limit")
        self.assertLess(gate.limit, grown)
        self.assertEqual(gate.decreases, 1)

    def test_timeout_sheds_load_without_taking_a_permit(self):
        gate = AdaptiveConcurrency(initial=1, minimum=1, maximum=1)
        self.assertTrue(gate.acquire(timeout=1))
        self.assertFalse(gate.acquire(timeout=0.05))
        self.assertEqual(gate.in_flight, 1)
        self.assertEqual(gate.shed, 1)

    def test_neutral_release_does_not_feed_pressure(self):
        gate = AdaptiveConcurrency(initial=8, minimum=2, maximum=32)
        self.assertTrue(gate.acquire(timeout=1))
        gate.release(neutral=True)
        self.assertEqual(gate.limit, 8)
        self.assertEqual(gate.failed, 0)
        self.assertEqual(gate.decreases, 0)

    def test_limits_are_clamped(self):
        gate = AdaptiveConcurrency(initial=100, minimum=4, maximum=8)
        self.assertEqual(gate.limit, 8)
        gate = AdaptiveConcurrency(initial=0, minimum=3, maximum=6)
        self.assertEqual(gate.limit, 3)


class ConfigDefaultsTest(unittest.TestCase):

    def test_latency_defaults_are_tuned(self):
        cfg = Config()
        self.assertLessEqual(cfg.queue_wait_sec, 7.0)
        self.assertGreaterEqual(cfg.initial_concurrent_requests, 8)
        self.assertLessEqual(cfg.initial_concurrent_requests, cfg.max_concurrent_requests)
        self.assertGreater(cfg.max_concurrent_requests, 8)
        self.assertGreater(cfg.hedge_after_sec, 0)
        self.assertGreater(cfg.first_byte_timeout_sec, 0)
        self.assertGreater(cfg.request_deadline_sec, 0)
        self.assertTrue(cfg.adaptive_concurrency)

    def test_env_overrides_new_knobs(self):
        from gemini_web2api.config import _apply_env_overrides
        cfg = Config()
        _apply_env_overrides(cfg, {
            "HEDGE_AFTER_SEC": "1.5",
            "FIRST_BYTE_TIMEOUT_SEC": "9",
            "ADAPTIVE_CONCURRENCY": "false",
            "MAX_CONCURRENT_REQUESTS": "6",
        })
        self.assertEqual(cfg.hedge_after_sec, 1.5)
        self.assertEqual(cfg.first_byte_timeout_sec, 9.0)
        self.assertFalse(cfg.adaptive_concurrency)
        self.assertEqual(cfg.max_concurrent_requests, 6)
        self.assertLessEqual(cfg.initial_concurrent_requests, 6)

    def test_config_bool_strings_parse_falsey(self):
        # "false"/"0" in config.json must disable the flag, not enable it
        from gemini_web2api.config import _apply
        cfg = Config()
        _apply(cfg, {"adaptive_concurrency": "false", "agents_enabled": "0",
                     "temporary_chats": True, "log_requests": "off"})
        self.assertFalse(cfg.adaptive_concurrency)
        self.assertFalse(cfg.agents_enabled)
        self.assertFalse(cfg.log_requests)
        self.assertTrue(cfg.temporary_chats)


# ---------------------------------------------------------------------------
# Tail-latency hedging
# ---------------------------------------------------------------------------

def bare_client(**overrides):
    return GeminiClient(Config(**overrides))


class HedgeRaceTest(unittest.TestCase):

    def test_stalled_primary_is_raced_by_hedge(self):
        client = bare_client(hedge_after_sec=0.05, first_byte_timeout_sec=2.0,
                             request_deadline_sec=5.0, stream_read_timeout_sec=1.0,
                             hedge_max_inflight=2)
        client.limiter = AdaptiveConcurrency(initial=2, minimum=1, maximum=4)
        calls = []

        def fake_post(inner, cancel_event=None, response_box=None, timeout=None):
            calls.append(time.monotonic())
            if len(calls) == 1:
                while not cancel_event.is_set():
                    time.sleep(0.005)
                return
            yield b"HEDGE-LINE"
            time.sleep(0.01)
            yield b"SECOND-LINE"

        client._post_stream_once = fake_post
        started = time.monotonic()
        lines = list(client._hedged_lines("inner", time.monotonic() + 5))
        elapsed = time.monotonic() - started
        self.assertEqual(lines, [b"HEDGE-LINE", b"SECOND-LINE"])
        self.assertEqual(len(calls), 2)
        self.assertLess(elapsed, 1.0)  # not the primary's infinite stall
        # the hedge permit was returned neutrally
        self.assertEqual(client.limiter.in_flight, 0)
        self.assertEqual(client.limiter.failed, 0)

    def test_hedge_is_refused_when_gate_is_saturated(self):
        client = bare_client(hedge_after_sec=0.05, first_byte_timeout_sec=0.4,
                             request_deadline_sec=5.0, stream_read_timeout_sec=1.0,
                             hedge_max_inflight=2)
        gate = AdaptiveConcurrency(initial=1, minimum=1, maximum=1)
        client.limiter = gate
        self.assertTrue(gate.acquire(timeout=1))  # the client request's own slot

        def fake_post(inner, cancel_event=None, response_box=None, timeout=None):
            while not cancel_event.is_set():
                time.sleep(0.005)
            return
            yield b""  # pragma: no cover

        client._post_stream_once = fake_post
        started = time.monotonic()
        with self.assertRaises(RetryableError) as ctx:
            list(client._hedged_lines("inner", time.monotonic() + 5))
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(ctx.exception.kind, "stall")
        self.assertEqual(gate.in_flight, 1)  # caller's permit untouched

    def test_first_byte_guard_trips_without_hedging(self):
        client = bare_client(hedge_after_sec=0, first_byte_timeout_sec=0.2,
                             request_deadline_sec=5.0, stream_read_timeout_sec=1.0)
        client.limiter = AdaptiveConcurrency(initial=2, minimum=1, maximum=4)

        def fake_post(inner, cancel_event=None, response_box=None, timeout=None):
            while not cancel_event.is_set():
                time.sleep(0.005)
            return
            yield b""  # pragma: no cover

        client._post_stream_once = fake_post
        started = time.monotonic()
        with self.assertRaises(RetryableError) as ctx:
            list(client._hedged_lines("inner", time.monotonic() + 5))
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(ctx.exception.kind, "stall")


# ---------------------------------------------------------------------------
# Live server: burst absorption, fast fail, keep-alive
# ---------------------------------------------------------------------------

class ServerLatencyTest(unittest.TestCase):

    def test_burst_is_absorbed_with_adaptive_growth(self):
        fake = SlowFakeGemini(text="abcdefghijklmnop", delay=0.01)
        app = make_app(fake, initial_concurrent_requests=2,
                       max_concurrent_requests=8, queue_wait_sec=5.0,
                       request_deadline_sec=30.0)
        server = make_server(app)
        try:
            port = server.server_address[1]
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(request, port, "POST", "/v1/chat/completions", CHAT)
                           for _ in range(8)]
                results = [f.result(timeout=20) for f in futures]
            elapsed = time.monotonic() - started
            self.assertEqual([r[0] for r in results], [200] * 8)
            self.assertLess(elapsed, 5.0)
            self.assertGreater(app.limiter.peak_limit, 2)
            self.assertEqual(app.limiter.shed, 0)
        finally:
            server.shutdown()

    def test_overload_returns_fast_429_with_retry_after(self):
        app = make_app(SlowFakeGemini(), initial_concurrent_requests=1,
                       max_concurrent_requests=1, queue_wait_sec=0.25)
        server = make_server(app)
        try:
            port = server.server_address[1]
            self.assertTrue(app.limiter.acquire(timeout=1))
            try:
                started = time.monotonic()
                status, body, headers = request(port, "POST", "/v1/chat/completions", CHAT)
                elapsed = time.monotonic() - started
            finally:
                app.limiter.release()
            self.assertEqual(status, 429)
            self.assertEqual(json.loads(body)["error"]["code"], "server_busy")
            self.assertIn("Retry-After", headers)
            self.assertLess(elapsed, 1.5)  # shed fast, never a 17s pile-up
        finally:
            server.shutdown()

    def test_cancelled_upstream_maps_to_504(self):
        app = make_app(BoomGemini())
        server = make_server(app)
        try:
            status, body, headers = request(server.server_address[1], "POST",
                                            "/v1/chat/completions", CHAT)
            self.assertEqual(status, 504)
            self.assertEqual(json.loads(body)["error"]["type"], "timeout_error")
            self.assertIn("Retry-After", headers)
        finally:
            server.shutdown()

    def test_unknown_post_route_skips_gate_and_metrics(self):
        app = make_app(SlowFakeGemini())
        server = make_server(app)
        try:
            status, _, _ = request(server.server_address[1], "POST", "/nope", CHAT)
            self.assertEqual(status, 404)
            load = app.load_snapshot()
            self.assertEqual(load["in_flight"], 0)
            self.assertEqual(load["requests"], 0)  # not counted as a failure
        finally:
            server.shutdown()

    def test_health_exposes_load_and_latency(self):
        app = make_app(SlowFakeGemini())
        server = make_server(app)
        try:
            status, body, _ = request(server.server_address[1], "GET", "/")
            self.assertEqual(status, 200)
            load = json.loads(body)["load"]
            for key in ("limit", "in_flight", "waiting", "ttfb_ms", "latency_ms",
                        "queue_wait_sec", "requests", "failures"):
                self.assertIn(key, load)
        finally:
            server.shutdown()

    def test_sse_keepalive_comments_during_silent_upstream(self):
        fake = SlowFirstByteGemini(text="ok answer", delay=0.5)
        app = make_app(fake, sse_keepalive_sec=0.15)
        server = make_server(app)
        try:
            payload = dict(CHAT, stream=True)
            status, body, _ = request(server.server_address[1], "POST",
                                      "/v1/chat/completions", payload, timeout=20)
            self.assertEqual(status, 200)
            self.assertIn(b": keep-alive", body)
            chunks = [json.loads(line[6:]) for line in body.decode().splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            text = "".join(c["choices"][0]["delta"].get("content", "")
                           for c in chunks if c.get("choices"))
            self.assertEqual(text, fake.text)
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
