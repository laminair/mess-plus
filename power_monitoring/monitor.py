import json
import time
import psutil
import os
import subprocess

try:
    from jtop import jtop, JtopException
except ImportError:
    jtop = None

from threading import Thread, Event

import wandb.errors


class HWMonitor(Thread):

    do_run = True

    def __init__(self, monitoring_freq: float = 1.0, stop_event=None):
        """
        :param monitoring_freq: Monitoring frequency in samples per second (float)
        """
        super(HWMonitor, self).__init__(daemon=True)
        self.monitoring_freq = monitoring_freq
        self.stop_monitor = stop_event

        self.psutil_monitor = SystemStats()
        self.stats = {}

    def join(self, timeout: float = 1.0) -> None:
        self.psutil_monitor.jetson.close()
        super().join(timeout=timeout)

    def run(self):
        while self.stop_monitor is None or not self.stop_monitor.is_set():
            sys_stats = self.psutil_monitor.read()

            all_res = {**sys_stats}
            
            self.stats = all_res
            wandb.log(all_res)
            time.sleep(1 / self.monitoring_freq)
        return


class SystemStats(object):

    def __init__(self):
        super().__init__()

        self.disk_read_sys_mb, self.disk_write_sys_mb = 0, 0
        self.net_sent_sys_mb, self.net_recv_sys_mb = 0, 0
        self.bandwidth_snapshot_time_s = 0
        self.create_bandwidth_snapshot()

        self.power_dict = {
            "power/total": 0,
            "power/cpu": 0,
            "power/gpu": 0,
        }

        if jtop is not None:
            self.jetson = jtop()
            self.jetson.attach(self.read_power_metrics)
            try:
                self.jetson.start()
            except JtopException:
                pass

    @staticmethod
    def get_static_sys_info():
        return {
            "cpu/logical_core_count": psutil.cpu_count(logical=True),
            "memory/total_memory_sys_mb": psutil.virtual_memory().total / 1024 ** 2
        }

    def read(self):
        """
        Get the current system and process info of the Python runtime.
        Bandwidths are calculate over the last interval since this method was called
        """
        cpu_info = self.get_cpu_info()
        memory_info = self.get_memory_info()
        proc_info = self.get_process_info()
        disk_info = self.get_disk_info()
        net_info = self.get_network_info()
        bandwidth_info = self.get_bandwidths(disk_info=disk_info, net_info=net_info)

        if jtop is None:
            self.get_power_info_nvidia_smi()

        return {**cpu_info, **memory_info, **proc_info, **disk_info, **net_info, **bandwidth_info, **self.power_dict}

    def create_bandwidth_snapshot(self, disk_info=None, net_info=None):
        """
        Sets the disk and network counters + time to calculate the bandwidth on the next call of `get_bandwidths`
        """
        if disk_info is None:
            disk_info = self.get_disk_info()
        self.disk_read_sys_mb = disk_info["disk/disk_read_sys_mb"]
        self.disk_write_sys_mb = disk_info["disk/disk_write_sys_mb"]

        if net_info is None:
            net_info = self.get_network_info()
        self.net_sent_sys_mb = net_info["network/net_sent_sys_mb"]
        self.net_recv_sys_mb = net_info["network/net_recv_sys_mb"]
        self.bandwidth_snapshot_s = time.time()

    def get_bandwidths(self, disk_info, net_info):
        """
        Calculate the difference between the disk and network read/written bytes since the last call
        Populates the member variables that cached the last state + time
        """

        old_disk_read_sys_mb = self.disk_read_sys_mb
        old_disk_write_sys_mb = self.disk_write_sys_mb
        old_net_sent_sys_mb = self.net_sent_sys_mb
        old_net_recv_sys_mb = self.net_recv_sys_mb
        old_bandwidth_snapshot_s = self.bandwidth_snapshot_s

        self.create_bandwidth_snapshot()

        disk_read_sys_timeframe_mb = self.disk_read_sys_mb - old_disk_read_sys_mb
        disk_write_sys_timeframe_mb = self.disk_write_sys_mb - old_disk_write_sys_mb
        net_sent_sys_timeframe_mb = self.net_sent_sys_mb - old_net_sent_sys_mb
        net_recv_sys_timeframe_mb = self.net_recv_sys_mb - old_net_recv_sys_mb
        time_diff_s = self.bandwidth_snapshot_s - old_bandwidth_snapshot_s

        disk_read_sys_bandwidth_mbs = disk_read_sys_timeframe_mb / time_diff_s
        disk_write_sys_bandwidth_mbs = disk_write_sys_timeframe_mb / time_diff_s
        net_sent_sys_bandwidth_mbs = net_sent_sys_timeframe_mb / time_diff_s
        net_recv_sys_bandwidth_mbs = net_recv_sys_timeframe_mb / time_diff_s

        return {
            "bandwidth/disk_read_sys_bandwidth_mbs": disk_read_sys_bandwidth_mbs,
            "bandwidth/disk_write_sys_bandwidth_mbs": disk_write_sys_bandwidth_mbs,
            "bandwidth/net_sent_sys_bandwidth_mbs": net_sent_sys_bandwidth_mbs,
            "bandwidth/net_recv_sys_bandwidth_mbs": net_recv_sys_bandwidth_mbs
        }

    @staticmethod
    def get_cpu_info():
        # hyperthreaded cores included
        # type: int
        logical_core_count = psutil.cpu_count(logical=True)

        # global cpu stats, ever-increasing from boot
        # type: (int, int, int, int)
        cpu_stats = psutil.cpu_stats()

        # average system load over 1, 5 and 15 minutes summarized over all cores in percent
        # type: (float, float, float)
        one_min, five_min, fifteen_min = psutil.getloadavg()
        avg_sys_load_one_min_percent = one_min / logical_core_count * 100
        avg_sys_load_five_min_percent = five_min / logical_core_count * 100
        avg_sys_load_fifteen_min_percent = fifteen_min / logical_core_count * 100

        return {
            "cpu/interrupts/global_ctx_switches_count": cpu_stats.ctx_switches,
            "cpu/interrupts/global_interrupts_count": cpu_stats.interrupts,
            "cpu/interrupts/global_soft_interrupts_count": cpu_stats.soft_interrupts,
            "cpu/load/avg_sys_load_one_min_percent": avg_sys_load_one_min_percent,
            "cpu/load/avg_sys_load_five_min_percent": avg_sys_load_five_min_percent,
            "cpu/load/avg_sys_load_fifteen_min_percent": avg_sys_load_fifteen_min_percent
        }

    @staticmethod
    def get_memory_info():

        # global memory information
        # type (int): total_b - total memory on the system in bytes
        # type (int): available_b - available memory on the system in bytes
        # type (float): used_percent - total / used_b
        # type (int): used_b - used memory on the system in bytes (may not match "total - available" or "total - free")
        mem_stats = psutil.virtual_memory()

        total_memory_sys_mb = mem_stats.total / 1024 ** 2
        available_memory_sys_mb = mem_stats.available / 1024 ** 2
        used_memory_sys_mb = mem_stats.used / 1024 ** 2

        return {
            "memory/total_memory_sys_mb": total_memory_sys_mb,
            "memory/available_memory_sys_mb": available_memory_sys_mb,
            "memory/used_memory_sys_mb": used_memory_sys_mb,
            "memory/used_memory_sys_percent": used_memory_sys_mb
        }

    @staticmethod
    def get_process_info():

        # gets its own pid by default
        proc = psutil.Process()

        # voluntary and involunatry context switches by the process (cumulative)
        # type: (int, int)
        voluntary_proc_ctx_switches, involuntary_proc_ctx_switches = proc.num_ctx_switches()

        # memory information
        # type (int): rrs_b - resident set size: non-swappable physical memory used in bytes
        # type (int): vms_b - virtual memory size: total amount of virtual memory used in bytes
        # type (int): shared_b - shared memory size in bytes
        # type (int): trs_b - text resident set: memory devoted to executable code in bytes
        # type (int): drs_b - data resident set: physical memory devoted to other than code in bytes
        # type (int): lib_b - library memory: memory used by shared libraries in bytes
        # type (int): dirty_pages_count - number of dirty pages
        mem_info = proc.memory_info()

        # Memory info available on all platforms. See: psutil.readthedocs.io/en/latest/index.html?highlight=memory_info
        resident_set_size_proc_mb = mem_info.rss / 1024 ** 2
        virtual_memory_size_proc_mb = mem_info.vms / 1024 ** 2

        memory_dict = {
            "process/voluntary_proc_ctx_switches": voluntary_proc_ctx_switches,
            "process/involuntary_proc_ctx_switches": involuntary_proc_ctx_switches,
            "process/memory/resident_set_size_proc_mb": resident_set_size_proc_mb,
            "process/memory/virtual_memory_size_proc_mb": virtual_memory_size_proc_mb
        }

        try:
            # Linux attributes
            memory_dict["process/memory/shared_memory_proc_mb"] = mem_info.shared / 1024 ** 2
            memory_dict["process/memory/text_resident_set_proc_mb"] = mem_info.text / 1024 ** 2
            memory_dict["process/memory/data_resident_set_proc_mb"] = mem_info.data / 1024 ** 2
            memory_dict["process/memory/lib_memory_proc_mb"] = mem_info.lib / 1024 ** 2
        except AttributeError:
            pass

        return memory_dict

    @staticmethod
    def get_disk_info():

        # system disk stats
        # type (int): disk_read_sys_count - how often were reads performed
        # type (int): disk_write_sys_count - how often were writes performed
        # type (int): disk_read_sys_bytes - how much was read in bytes
        # type (int): writen_sys_bytes - how much was written in bytes
        # type (int): disk_read_time_sys_ms - how much time was used to read in milliseconds
        # type (int): disk_write_time_sys_ms - how much time was used to write in milliseconds
        # type (int): busy_time_sys_ms - how much time was used for actual I/O
        disk_info = psutil.disk_io_counters()

        disk_read_sys_mb = disk_info.read_bytes / 1024 ** 2
        disk_write_sys_mb = disk_info.write_bytes / 1024 ** 2
        disk_read_time_sys_s = disk_info.read_time / 1000
        disk_write_time_sys_s = disk_info.write_time / 1000

        disk_info_dict = {
            "disk/counter/disk_read_sys_count": disk_info.read_count,
            "disk/counter/disk_write_sys_count": disk_info.write_count,
            "disk/disk_read_sys_mb": disk_read_sys_mb,
            "disk/disk_write_sys_mb": disk_write_sys_mb,
            "disk/time/disk_read_time_sys_s": disk_read_time_sys_s,
            "disk/time/disk_write_time_sys_s": disk_write_time_sys_s
            # , "disk_busy_time_sys_s": disk_busy_time_sys_s
        }

        try:
            disk_info_dict["disk/time/disk_busy_time_sys_s"] = disk_info.busy_time / 1000  # returns seconds
        except AttributeError:
            pass

        return disk_info_dict

    @staticmethod
    def get_network_info():

        # network system stats
        # type (int): net_sent_sys_bytes - sent bytes over all network interfaces
        # type (int): net_recv_sys_bytes - received bytes over all network interfaces
        net_info = psutil.net_io_counters(pernic=False)

        return {
            "network/net_sent_sys_mb": net_info.bytes_sent / 1024 ** 2,
            "network/net_recv_sys_mb": net_info.bytes_recv / 1024 ** 2
        }

    def read_power_metrics(self, jetson):
        json_data = jetson.json()
        json_data = json.loads(json_data)

        self.power_dict.update({
            "power/total": json_data["power"]["tot"]["power"],
            "power/cpu": json_data["power"]["rail"]["VDD_CPU_CV"]["power"],
            "power/gpu": json_data["power"]["rail"]["VDD_GPU_SOC"]["power"],
        })

    def get_power_info_nvidia_smi(self):
        out = os.popen("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits").read().strip()
        try:
            power = float(out) * 1000  # We want to have power in mW (identical to jTOP).
        except ValueError:
            power = 0

        self.power_dict = {
            "power/total": power,
            "power/cpu": 0,
            "power/gpu": power,
        }


if __name__ == "__main__":
    mon = HWMonitor(monitoring_freq=1.0)
    mon.start()

    try:
        time.sleep(10)
        mon.stop_monitor.set()
    except KeyboardInterrupt:
        mon.join()