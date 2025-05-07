#!/usr/bin/env python3
import time
import psutil
from prometheus_client import start_http_server, Gauge, Counter, Summary

# Metrik sistem
CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percent')
MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'System memory usage in bytes')

# Contoh counter custom
HEARTBEAT = Counter('exporter_heartbeat_total', 'Number of heartbeats sent')

# Contoh summary custom
LOOP_DURATION = Summary('exporter_loop_duration_seconds', 'Time spent in main loop')

def collect_system_metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    MEMORY_USAGE.set(psutil.virtual_memory().used)

@LOOP_DURATION.time()
def main_loop():
    # Hitungan heartbeat
    HEARTBEAT.inc()
    # Ambil metrik CPU & memory
    collect_system_metrics()
    # Tidur sejenak
    time.sleep(5)

if __name__ == "__main__":
    # Jalankan HTTP server Prometheus di port 8001 (atau port lain bebas)
    start_http_server(8001)
    print("Prometheus exporter running on port 8001")
    # Loop selamanya
    while True:
        main_loop()
