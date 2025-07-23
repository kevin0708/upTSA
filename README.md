# upTSA
Time Series Analysis on UPMEM
# Time Series Analysis on UPMEM DPUs

An implementation of time series analysis algorithms leveraging UPMEM PIM architecture for motif discovery.

## Key Features
- 🚀 **Massive Parallelism**: 
  - 2530 DPUs (default) × 16 tasklets per DPU
- ⏱ **Efficiency**: 
  - Near-memory computation eliminates data transfer bottlenecks

## Project Structure
.
├── host.c # Host code (task dispatch/results collection)
├── dpu.c # DPU kernel (time series processing)
├── support/
│ ├── timer.h # Timer
│ └── params.h # Parameters
| └── common.h # Common Function
├── Makefile # Build automation


## Prerequisites
- [UPMEM SDK](https://sdk.upmem.com/) (2023.1+)
- GCC ≥9.0 (with C99 and OpenMP support)
- Recommended Hardware:
  - Server with UPMEM DIMMs
 
## Build & Run

### Compilation
```bash
make  # Default: 2530 DPUs, 16 tasklets/DPU
Custom configuration:
```

Execution
Process built-in test data:

bash
./ts_host
