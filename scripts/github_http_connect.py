#!/usr/bin/env python3
"""SSH ProxyCommand through an HTTP CONNECT proxy; credentials stay in ~/.ssh."""
import os
import select
import socket
import sys

def main():
    host, port = sys.argv[1], int(sys.argv[2])
    proxy_host = os.environ.get("SSH_HTTP_PROXY_HOST", "127.0.0.1")
    proxy_port = int(os.environ.get("SSH_HTTP_PROXY_PORT", "7897"))
    with socket.create_connection((proxy_host, proxy_port), timeout=15) as connection:
        request = f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n"
        connection.sendall(request.encode("ascii"))
        header = bytearray()
        while not header.endswith(b"\r\n\r\n"):
            value = connection.recv(1)
            if not value or len(header) > 16384:
                raise RuntimeError("Proxy closed connection or returned an oversized header")
            header.extend(value)
        status = bytes(header).split(b"\r\n", 1)[0].decode("ascii", errors="replace")
        if len(status.split()) < 2 or status.split()[1] != "200":
            raise RuntimeError("HTTP CONNECT rejected: " + status)
        connection.settimeout(None)
        inputs = [sys.stdin.buffer, connection]
        while True:
            readable, _, _ = select.select(inputs, [], [])
            if sys.stdin.buffer in readable:
                data = os.read(sys.stdin.fileno(), 65536)
                if data:
                    connection.sendall(data)
                else:
                    inputs.remove(sys.stdin.buffer)
                    connection.shutdown(socket.SHUT_WR)
            if connection in readable:
                data = connection.recv(65536)
                if not data:
                    return
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()

if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, IndexError) as error:
        print("SSH HTTP proxy: " + str(error), file=sys.stderr)
        sys.exit(1)
