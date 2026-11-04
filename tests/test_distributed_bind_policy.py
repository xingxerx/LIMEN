"""Bind policy for the distributed coordination server: loopback by default,
and cleartext gRPC only on loopback unless the operator opts in explicitly."""

import os
import unittest
from unittest import mock

from limen.distributed.config import (
    ALLOW_INSECURE_NETWORK_ENV,
    InsecureBindError,
    NodeConfig,
    default_node_host,
    is_loopback_host,
    is_tailnet_host,
)
from limen.distributed.server import serve


def _cfg(host: str, **kw) -> NodeConfig:
    return NodeConfig(node_id="n", host=host, port=0, **kw)


class LoopbackTests(unittest.TestCase):
    def test_loopback_addresses_are_loopback(self):
        for host in ("127.0.0.1", "127.5.5.5", "::1", "[::1]", "localhost", "LOCALHOST"):
            self.assertTrue(is_loopback_host(host), host)

    def test_everything_else_is_network_reachable(self):
        for host in ("0.0.0.0", "::", "", "192.168.1.5", "10.0.0.2", "100.64.0.1", "example.com"):
            self.assertFalse(is_loopback_host(host), host)


class TailnetTests(unittest.TestCase):
    def test_the_tailscale_range_is_recognised(self):
        for host in ("100.64.0.1", "100.106.147.75", "100.127.255.255", "[100.100.100.100]"):
            self.assertTrue(is_tailnet_host(host), host)

    def test_neighbouring_and_non_ip_hosts_are_not(self):
        for host in ("100.63.255.255", "100.128.0.0", "192.168.1.5", "::1", "example.com", ""):
            self.assertFalse(is_tailnet_host(host), host)


class BindPolicyTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(ALLOW_INSECURE_NETWORK_ENV, None)

    def test_loopback_may_run_without_tls(self):
        _cfg("127.0.0.1").check_bind_policy()

    def test_wildcard_or_lan_without_tls_is_refused(self):
        for host in ("0.0.0.0", "::", "192.168.1.5", "node-b.example.com"):
            with self.assertRaises(InsecureBindError, msg=host):
                _cfg(host).check_bind_policy()

    def test_the_refusal_names_the_ways_out(self):
        with self.assertRaises(InsecureBindError) as ctx:
            _cfg("0.0.0.0").check_bind_policy()
        message = str(ctx.exception)
        self.assertIn("LIMEN_TLS_CERT", message)
        self.assertIn(ALLOW_INSECURE_NETWORK_ENV, message)

    def test_a_tailnet_address_without_tls_is_refused(self):
        with self.assertRaises(InsecureBindError):
            _cfg("100.106.147.75").check_bind_policy()

    def test_a_tailnet_address_with_mutual_tls_is_allowed_silently(self):
        cfg = _cfg("100.106.147.75", tls_cert_path="c.pem", tls_key_path="k.pem", tls_ca_path="ca.pem")
        with self.assertNoLogs("limen.distributed", level="WARNING"):
            cfg.check_bind_policy()

    def test_the_tailscale_default_without_tls_fails_with_a_way_out(self):
        env = {"LIMEN_NODE_ID": "n"}
        with mock.patch("limen.distributed.config._tailscale_ipv4", return_value="100.106.147.75"):
            with mock.patch.dict(os.environ, env, clear=True):
                cfg = NodeConfig.from_env()
        with self.assertRaises(InsecureBindError) as ctx:
            cfg.check_bind_policy()
        self.assertIn("loopback", str(ctx.exception))

    def test_tls_allows_a_network_address_but_warns_without_a_client_ca(self):
        cfg = _cfg("0.0.0.0", tls_cert_path="c.pem", tls_key_path="k.pem")
        with self.assertLogs("limen.distributed", level="WARNING") as logs:
            cfg.check_bind_policy()
        self.assertIn("without LIMEN_TLS_CA", logs.output[0])

    def test_mutual_tls_is_silent(self):
        cfg = _cfg("0.0.0.0", tls_cert_path="c.pem", tls_key_path="k.pem", tls_ca_path="ca.pem")
        with self.assertNoLogs("limen.distributed", level="WARNING"):
            cfg.check_bind_policy()

    def test_a_certificate_without_a_key_is_not_tls(self):
        with self.assertRaises(InsecureBindError):
            _cfg("0.0.0.0", tls_cert_path="c.pem").check_bind_policy()

    def test_explicit_opt_in_allows_cleartext_and_logs_a_warning(self):
        os.environ[ALLOW_INSECURE_NETWORK_ENV] = "1"
        with self.assertLogs("limen.distributed", level="WARNING") as logs:
            _cfg("0.0.0.0").check_bind_policy()
        self.assertIn("cleartext", logs.output[0])

    def test_only_the_exact_opt_in_value_counts(self):
        for value in ("true", "yes", "0", ""):
            os.environ[ALLOW_INSECURE_NETWORK_ENV] = value
            with self.assertRaises(InsecureBindError, msg=value):
                _cfg("0.0.0.0").check_bind_policy()

    def test_serve_refuses_before_creating_or_binding_anything(self):
        with mock.patch("limen.distributed.server.grpc.server") as make_server:
            with self.assertRaises(InsecureBindError):
                serve(_cfg("0.0.0.0"), registry=object(), port=0)
        make_server.assert_not_called()


class FromEnvTests(unittest.TestCase):
    def test_default_host_is_loopback_without_tailscale(self):
        with mock.patch("limen.distributed.config._tailscale_ipv4", return_value=None):
            with mock.patch.dict(os.environ, {"LIMEN_NODE_ID": "n"}, clear=True):
                self.assertEqual(NodeConfig.from_env().host, "127.0.0.1")

    def test_default_host_is_the_tailscale_address_when_there_is_one(self):
        with mock.patch("limen.distributed.config._tailscale_ipv4", return_value="100.106.147.75"):
            self.assertEqual(default_node_host(), "100.106.147.75")
            with mock.patch.dict(os.environ, {"LIMEN_NODE_ID": "n"}, clear=True):
                self.assertEqual(NodeConfig.from_env().host, "100.106.147.75")

    def test_an_empty_host_variable_falls_back_to_the_default(self):
        with mock.patch("limen.distributed.config._tailscale_ipv4", return_value=None):
            env = {"LIMEN_NODE_ID": "n", "LIMEN_NODE_HOST": ""}
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(NodeConfig.from_env().host, "127.0.0.1")

    def test_an_explicit_host_is_honoured(self):
        env = {"LIMEN_NODE_ID": "n", "LIMEN_NODE_HOST": "10.1.2.3"}
        with mock.patch("limen.distributed.config._tailscale_ipv4", return_value="100.106.147.75"):
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(NodeConfig.from_env().host, "10.1.2.3")


if __name__ == "__main__":
    unittest.main()