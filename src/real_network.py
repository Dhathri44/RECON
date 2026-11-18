
import socket
import struct
import time
import hashlib

def receive_exact(sock, length):
    chunks = []
    remaining = length

    while remaining:
        data = sock.recv(remaining)

        if not data:
            raise ConnectionError("Connection closed unexpectedly")

        chunks.append(data)
        remaining -= len(data)

    return b"".join(chunks)

def send_file(host, port, file_path):
    with open(file_path, "rb") as f:
        payload = f.read()

    digest = hashlib.sha256(payload).digest()

    start = time.perf_counter()

    with socket.create_connection((host, port)) as sock:
        sock.sendall(struct.pack("!Q", len(payload)))
        sock.sendall(digest)
        sock.sendall(payload)

        response = receive_exact(sock, 32)

    elapsed = time.perf_counter() - start

    if response != digest:
        raise RuntimeError("Receiver hash does not match")

    throughput = (
        len(payload) * 8 / elapsed / 1_000_000
        if elapsed > 0 else 0
    )

    print("Transfer successful")
    print(f"Bytes transferred: {len(payload)}")
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print(f"Throughput: {throughput:.4f} Mbps")

def receive_file(host, port, output_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)

        print(f"Listening on {host}:{port}")

        conn, address = server.accept()

        with conn:
            print("Connected:", address)

            length = struct.unpack(
                "!Q", receive_exact(conn, 8)
            )[0]

            digest = receive_exact(conn, 32)
            payload = receive_exact(conn, length)

            actual_digest = hashlib.sha256(payload).digest()

            if actual_digest != digest:
                raise RuntimeError("File integrity verification failed")

            with open(output_path, "wb") as f:
                f.write(payload)

            conn.sendall(actual_digest)

            print(f"Received {length} bytes")
            print("File integrity verified")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["send", "receive"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--file", required=True)

    args = parser.parse_args()

    if args.mode == "send":
        send_file(args.host, args.port, args.file)
    else:
        receive_file(args.host, args.port, args.file)