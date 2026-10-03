#!/usr/bin/env python3
import socket
import struct
import sys

def probe(ip, port, timeout=3.0):
    magic = bytes([0x00,0xFF,0xFF,0x00,0xFE,0xFE,0xFE,0xFE,0xFD,0xFD,0xFD,0xFD,0x12,0x34,0x56,0x78])
    packet = bytes([0x01]) + struct.pack(">q", 0) + magic + struct.pack(">q", 12345)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        sock.sendto(packet, (ip, port))
        data, addr = sock.recvfrom(2048)
        print(f"[OK] {ip}:{port} -> reply from {addr}, {len(data)} bytes")
        return True
    except socket.timeout:
        print(f"[FAIL] {ip}:{port} -> no reply within {timeout}s")
        return False
    finally:
        sock.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <ip> <port>")
        sys.exit(1)
    probe(sys.argv[1], int(sys.argv[2]))
