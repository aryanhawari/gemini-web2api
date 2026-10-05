"""Agent-fleet tests: lanes, parallel execution, supervision, overload.

All upstream work is faked — these tests exercise the lane classifier, the
lane-scheduled agent threads, the crash supervisor and the server's fast
overload signal while the fleet is saturated.
"""

import json
import os
import sys
import threading
import time
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_web2api.agents import (EXPRESS, HEAVY, STANDARD, AgentError, AgentPool,
                                   PoolBusy, build_pool, classify, payload_has_media)
from gemini_web2api.config import Config
from gemini_web2api.server import App, make_handler


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------

class InstantGemini:
    """Answers immediately; enough for lane/overload tests."""

    def generate(self, prompt, file_refs=None, model_info=None):
        return "ok"

    def stream_events(self, prompt, file_refs=None, model_info=None):
        yield ("text", "ok")

    def stream_generate(self, prompt, file_refs=None, model_info=None):
        yield "ok"


class NoopUploader:
    def upload(self, data, mime):
        return "/uploaded/ref"


def make_app(fake=None, **overrides):
    config = Config(**overrides)
    return App(config, gemini=fake or InstantGemini(), uploader=NoopUploader(),
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


def wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


CHAT = {"model": "gemini-3.6-flash",
        "messages": [{"role": "user", "content": "hi"}]}


# ---------------------------------------------------------------------------
# Lane classification
# ---------------------------------------------------------------------------

class ClassifyTest(unittest.TestCase):

    def test_small_body_is_express(self):
        self.assertEqual(classify(80, CHAT), EXPRESS)

    def test_mid_body_is_standard(self):
        self.assertEqual(classify(15000, CHAT), STANDARD)

    def test_large_body_is_heavy(self):
        self.assertEqual(classify(80000, CHAT), HEAVY)

    def test_tools_force_heavy(self):
        body = dict(CHAT, tools=[{"type": "function",
                                  "function": {"name": "get_weather"}}])
        self.assertEqual(classify(200, body), HEAVY)

    def test_openai_image_forces_heavy(self):
        body = dict(CHAT, messages=[{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}])
        self.assertTrue(payload_has_media(body))
        self.assertEqual(classify(200, body), HEAVY)

    def test_google_inline_data_forces_heavy(self):
        body = {"contents": [{"role": "user", "parts": [
            {"text": "look"},
            {"inlineData": {"mimeType": "image/png", "data": "AAAA"}}]}]}
        self.assertEqual(classify(200, body), HEAVY)

    def test_plain_payload_has_no_media(self):
        self.assertFalse(payload_has_media(CHAT))

    def test_never_raises_on_weird_input(self):
        self.assertIn(classify(100, "not-a-dict"), (EXPRESS, STANDARD, HEAVY))
        self.assertIn(classify(None, None), (EXPRESS, STANDARD, HEAVY))


# ---------------------------------------------------------------------------
# Config knobs
# ---------------------------------------------------------------------------

class AgentConfigTest(unittest.TestCase):

    def test_default_fleet_is_ten_plus_agents(self):
        cfg = Config()
        pool = build_pool(cfg)
        self.assertIsNotNone(pool)
        describe = pool.describe()
        self.assertGreaterEqual(describe["agents"], 10)
        self.assertGreaterEqual(describe["lanes"][EXPRESS]["agents"], 1)
        self.assertGreaterEqual(describe["lanes"][STANDARD]["agents"], 1)
        self.assertGreaterEqual(describe["lanes"][HEAVY]["agents"], 1)
        self.assertEqual(
            cfg.agents_express + cfg.agents_standard + cfg.agents_heavy,
            describe["agents"])

    def test_disabled_fleet_builds_none(self):
        self.assertIsNone(build_pool(Config(agents_enabled=False)))

    def test_empty_fleet_builds_none(self):
        self.assertIsNone(build_pool(Config(agents_express=0, agents_standard=0,
                                            agents_heavy=0)))

    def test_env_overrides_agent_knobs(self):
        from gemini_web2api.config import _apply_env_overrides
        cfg = Config()
        _apply_env_overrides(cfg, {
            "AGENTS_ENABLED": "false",
            "AGENTS_EXPRESS": "0",
            "AGENTS_STANDARD": "4",
            "AGENT_SUBMIT_WAIT_SEC": "1.5",
        })
        self.assertFalse(cfg.agents_enabled)
        self.assertEqual(cfg.agents_express, 0)
        self.assertEqual(cfg.agents_standard, 4)
        self.assertEqual(cfg.agent_submit_wait_sec, 1.5)


# ---------------------------------------------------------------------------
# Pool: parallel execution, lanes, supervision
# ---------------------------------------------------------------------------

class AgentPoolTest(unittest.TestCase):

    def test_jobs_run_in_parallel_across_agents(self):
        pool = AgentPool(express=2, standard=2, heavy=2, queue_depth=16,
                         submit_wait_sec=2)
        try:
            gate = threading.Barrier(6, timeout=5)
            jobs = [pool.submit(EXPRESS, lambda: (gate.wait(), "ok")[1])
                    for _ in range(6)]
            results = [job.result_or_raise(timeout=10) for job in jobs]
            self.assertEqual(results, ["ok"] * 6)
        finally:
            pool.stop()

    def test_result_and_error_propagate(self):
        pool = AgentPool(express=2, standard=0, heavy=0, submit_wait_sec=2)
        try:
            self.assertEqual(pool.run(EXPRESS, lambda: 42), 42)

            def boom():
                raise ValueError("nope")

            with self.assertRaises(ValueError):
                pool.run(EXPRESS, boom)
            self.assertEqual(pool.describe()["failed"], 1)
        finally:
            pool.stop()

    def test_express_stays_fast_while_heavy_is_saturated(self):
        pool = AgentPool(express=2, standard=0, heavy=2, queue_depth=16,
                         submit_wait_sec=2)
        release = threading.Event()

        def blocker():
            release.wait(5)
            return "heavy"

        try:
            jobs = [pool.submit(HEAVY, blocker) for _ in range(4)]
            self.assertTrue(wait_until(
                lambda: pool.describe()["lanes"][HEAVY]["busy"] == 2))
            self.assertEqual(pool.describe()["lanes"][HEAVY]["queued"], 2)
            started = time.monotonic()
            self.assertEqual(pool.run(EXPRESS, lambda: "fast", timeout=1), "fast")
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            release.set()
            for job in locals().get("jobs", []):
                job.wait(5)
            pool.stop()

    def test_idle_standard_agents_help_the_heavy_lane(self):
        pool = AgentPool(express=0, standard=2, heavy=1, queue_depth=16,
                         submit_wait_sec=2)
        try:
            gate = threading.Barrier(3, timeout=5)
            jobs = [pool.submit(HEAVY, lambda: (gate.wait(), "h")[1])
                    for _ in range(3)]
            results = [job.result_or_raise(timeout=10) for job in jobs]
            self.assertEqual(results, ["h"] * 3)
        finally:
            pool.stop()

    def test_overload_sheds_fast_with_pool_busy(self):
        pool = AgentPool(express=1, standard=0, heavy=0, queue_depth=1,
                         submit_wait_sec=0)
        release = threading.Event()

        def blocker():
            release.wait(5)

        try:
            pool.submit(EXPRESS, blocker)
            self.assertTrue(wait_until(
                lambda: pool.describe()["lanes"][EXPRESS]["busy"] == 1))
            pool.submit(EXPRESS, blocker)  # fills the single queue slot
            started = time.monotonic()
            with self.assertRaises(PoolBusy):
                pool.submit(EXPRESS, blocker, timeout=0.2)
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 1.0)
            self.assertGreaterEqual(pool.describe()["shed"], 1)
        finally:
            release.set()
            pool.stop()

    def test_supervisor_revives_crashed_agent_and_fails_orphan_job(self):
        pool = AgentPool(express=1, standard=0, heavy=0, queue_depth=4,
                         submit_wait_sec=1)
        agent = pool.agents[0]
        original = agent._execute
        try:
            def crash(job):
                raise RuntimeError("agent bug")

            agent._execute = crash
            started = time.monotonic()
            with self.assertRaises(AgentError):
                pool.run(EXPRESS, lambda: "doomed", timeout=1)
            self.assertLess(time.monotonic() - started, 3.0)  # no hang
            self.assertTrue(wait_until(
                lambda: pool.describe()["restarts"] >= 1, timeout=5))
            self.assertTrue(wait_until(lambda: agent.alive, timeout=5))
            agent._execute = original
            self.assertEqual(pool.run(EXPRESS, lambda: "recovered", timeout=2),
                             "recovered")
            self.assertIsNotNone(pool.describe()["last_error"])
        finally:
            agent._execute = original
            pool.stop()

    def test_stop_fails_queued_jobs_instead_of_hanging(self):
        pool = AgentPool(express=1, standard=0, heavy=0, queue_depth=4,
                         submit_wait_sec=1)
        release = threading.Event()

        def blocker():
            release.wait(5)

        pool.submit(EXPRESS, blocker)
        self.assertTrue(wait_until(
            lambda: pool.describe()["lanes"][EXPRESS]["busy"] == 1))
        queued = pool.submit(EXPRESS, blocker)
        pool.stop()
        release.set()
        self.assertTrue(queued.wait(2))
        self.assertIsInstance(queued.error, AgentError)


# ---------------------------------------------------------------------------
# Live server: lane dispatch, health, fast overload
# ---------------------------------------------------------------------------

class ServerAgentTest(unittest.TestCase):

    def _serve(self, app):
        server = make_server(app)
        self.addCleanup(server.shutdown)
        self.addCleanup(app.shutdown)
        return server.server_address[1]

    def test_health_reports_agent_fleet(self):
        app = make_app(agents_express=2, agents_standard=3, agents_heavy=2)
        port = self._serve(app)
        status, body, _ = request(port, "GET", "/")
        self.assertEqual(status, 200)
        fleet = json.loads(body)["load"]["agents"]
        self.assertTrue(fleet["enabled"])
        self.assertFalse(fleet["running"])  # lazy: no POST served yet
        self.assertEqual(fleet["agents"], 7)
        self.assertEqual(fleet["lanes"][EXPRESS]["agents"], 2)
        self.assertEqual(fleet["lanes"][HEAVY]["agents"], 2)

    def test_express_chat_survives_saturated_heavy_lane(self):
        app = make_app(agents_express=2, agents_standard=0, agents_heavy=1,
                       agent_queue_depth=8, agent_submit_wait_sec=1)
        port = self._serve(app)
        release = threading.Event()

        def blocker():
            release.wait(5)
            return "heavy"

        try:
            app.agents.submit(HEAVY, blocker)
            self.assertTrue(wait_until(
                lambda: app.agents.describe()["lanes"][HEAVY]["busy"] == 1))
            started = time.monotonic()
            status, body, _ = request(port, "POST", "/v1/chat/completions", CHAT)
            elapsed = time.monotonic() - started
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "ok")
            self.assertLess(elapsed, 1.0)
        finally:
            release.set()

    def test_saturated_lane_returns_fast_429_with_retry_after(self):
        app = make_app(agents_express=1, agents_standard=0, agents_heavy=0,
                       agent_queue_depth=1, agent_submit_wait_sec=0.25)
        port = self._serve(app)
        release = threading.Event()

        def blocker():
            release.wait(5)

        try:
            app.agents.submit(EXPRESS, blocker)
            self.assertTrue(wait_until(
                lambda: app.agents.describe()["lanes"][EXPRESS]["busy"] == 1))
            app.agents.submit(EXPRESS, blocker)  # fills the queue slot
            started = time.monotonic()
            status, body, headers = request(port, "POST", "/v1/chat/completions", CHAT)
            elapsed = time.monotonic() - started
            self.assertEqual(status, 429)
            self.assertEqual(json.loads(body)["error"]["code"], "server_busy")
            self.assertIn("Retry-After", headers)
            self.assertLess(elapsed, 1.5)
        finally:
            release.set()

    def test_agents_disabled_runs_inline(self):
        app = make_app(agents_enabled=False)
        self.assertIsNone(app.agents)
        port = self._serve(app)
        status, body, _ = request(port, "POST", "/v1/chat/completions", CHAT)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "ok")
        self.assertIsNone(json.loads(
            request(port, "GET", "/")[1])["load"]["agents"].get("lanes"))


if __name__ == "__main__":
    unittest.main()
