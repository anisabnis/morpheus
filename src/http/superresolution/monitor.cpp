#include "monitor.h"
#include <sys/sysinfo.h>

extern "C" ngx_uint_t monitor_get_min_gpu_utilization();
extern "C" ngx_uint_t monitor_get_min_memory_utilization();
extern "C" ngx_uint_t monitor_get_cpu_utilization();

ngx_uint_t monitor_get_min_gpu_utilization(){

  NVML nvml;
  NVMLDeviceManager device_manager{nvml};

  const auto devices_begin = device_manager.devices_begin();
  const auto devices_end = device_manager.devices_end();

  ngx_uint_t min_gpu_utilization = 100;
  
  for (auto device = devices_begin; device != devices_end; ++device) {
    (*device).refresh_metrics_or_halt();
    const auto& info = (*device).get_info();
    if (info.metrics.gpu_utilization < min_gpu_utilization){
      min_gpu_utilization = info.metrics.gpu_utilization;
    }
  }

  return min_gpu_utilization;
}


ngx_uint_t monitor_get_min_memory_utilization(){

  NVML nvml;
  NVMLDeviceManager device_manager{nvml};

  const auto devices_begin = device_manager.devices_begin();
  const auto devices_end = device_manager.devices_end();

  ngx_uint_t min_memory_utilization = 100;
  
  for (auto device = devices_begin; device != devices_end; ++device) {
    (*device).refresh_metrics_or_halt();
    const auto& info = (*device).get_info();
    if (info.metrics.memory_utilization < min_memory_utilization){
      min_memory_utilization = info.metrics.memory_utilization;
    }
  }

  return min_memory_utilization;  
}


ngx_uint_t monitor_get_cpu_utilization(){
  struct sysinfo sys_info;
  
  if (sysinfo(&sys_info) == -1) {
    return -1;
  }

  ngx_uint_t cpu_usage =  sys_info.loads[0] / (1 << SI_LOAD_SHIFT) * 100;  
  return cpu_usage;  
}
