"""String-topic access to ROS 1 over rosbridge (roslibpy).

rosbridge instead of rospy lets the voice agent run wherever the microphone
and speaker are: on the Ubuntu 24.04 host (Python 3.12, WebRTC echo
canceller) as well as inside the Noetic container. mobipick_gpt reaches ROS
the same way, and gpt_robot_demo.launch already starts rosbridge on :9090.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict

_LOG = logging.getLogger(__name__)


class RosStringBridge:
    """Publish and subscribe ``std_msgs/String`` (and ``Bool``) topics."""

    def __init__(self, url: str = "ws://localhost:9090", connect_timeout: float = 15.0) -> None:
        import roslibpy

        self._roslibpy = roslibpy
        host, port = _split_url(url)
        self.ros = roslibpy.Ros(host=host, port=port)
        try:
            self.ros.run(timeout=connect_timeout)
        except Exception as exc:
            raise ConnectionError(f"could not connect to rosbridge at {url}: {exc}") from exc
        if not self.ros.is_connected:
            raise ConnectionError(f"could not connect to rosbridge at {url}")
        self._pubs: Dict[str, object] = {}
        self._subs = []
        self._lock = threading.Lock()
        _LOG.info("connected to rosbridge %s", url)

    def publish(self, topic: str, text: str) -> None:
        self._publisher(topic, "std_msgs/String").publish(self._roslibpy.Message({"data": text}))

    def publish_bool(self, topic: str, value: bool) -> None:
        self._publisher(topic, "std_msgs/Bool").publish(self._roslibpy.Message({"data": bool(value)}))

    def _publisher(self, topic: str, msg_type: str):
        with self._lock:
            pub = self._pubs.get(topic)
            if pub is None:
                pub = self._roslibpy.Topic(self.ros, topic, msg_type, latch=False, queue_size=10)
                pub.advertise()
                self._pubs[topic] = pub
                time.sleep(0.2)  # give subscribers a moment to connect to the new publisher
            return pub

    def subscribe(self, topic: str, callback: Callable[[str], None]) -> None:
        sub = self._roslibpy.Topic(self.ros, topic, "std_msgs/String")
        sub.subscribe(lambda msg: callback(str(msg.get("data", ""))))
        self._subs.append(sub)

    def subscribe_bool(self, topic: str, callback: Callable[[bool], None]) -> None:
        sub = self._roslibpy.Topic(self.ros, topic, "std_msgs/Bool")
        sub.subscribe(lambda msg: callback(bool(msg.get("data", False))))
        self._subs.append(sub)

    def close(self) -> None:
        for sub in self._subs:
            try:
                sub.unsubscribe()
            except Exception:
                pass
        for pub in self._pubs.values():
            try:
                pub.unadvertise()
            except Exception:
                pass
        try:
            self.ros.terminate()
        except Exception:
            pass


def _split_url(url: str) -> "tuple[str, int]":
    rest = url.split("://", 1)[-1].rstrip("/")
    host, _, port = rest.partition(":")
    return host or "localhost", int(port or 9090)


def rosbridge_url(default: str = "ws://localhost:9090") -> str:
    import os

    return os.environ.get("ROSBRIDGE_URL", default)

