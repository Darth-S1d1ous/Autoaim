# DXGI Desktop Duplication Failure When CUDA/TensorRT DLLs Are Present

## Symptom

`AutoaimApp` fails at startup with:

```
Duplicate output failed! HRESULT=0x887A0004
[EntryPoint] Fatal error: [App] DXGICapture init failed
```

`HRESULT 0x887A0004` is `DXGI_ERROR_UNSUPPORTED`.  
`TestCapture` (the capture-only smoke test) runs correctly on the same machine.

## Root Cause

When the Windows loader starts `AutoaimApp.exe`, it resolves all imported DLLs
listed in the executable's import table **before** `main()` is reached. This
includes:

| DLL                     | Origin                       |
| ----------------------- | ---------------------------- |
| `cudart64_13.dll`       | CUDA 13 runtime              |
| `nvinfer_10.dll`        | TensorRT 10 inference engine |
| `nvinfer_plugin_10.dll` | TensorRT built-in plugins    |
| `nvonnxparser_10.dll`   | TensorRT ONNX parser         |

During this early load phase the NVIDIA driver calls each DLL's
`DllMain(DLL_PROCESS_ATTACH)`. The NVIDIA user-mode driver uses this
opportunity to:

1. **Register DXGI/D3D11 hooks** in the process, redirecting certain internal
   COM vtable entries so that CUDA can later interoperate with D3D resources.
2. **Select the discrete GPU** (the CUDA-capable device) as the "preferred"
   D3D11 adapter for the process.

As a result, by the time `DXGICapture::init()` calls
`IDXGIOutput1::DuplicateOutput()`, the DXGI runtime's internal state has been
altered. `DuplicateOutput` requires that the `ID3D11Device` passed to it was
created on the **same adapter that owns the output being duplicated**. Due to
the NVIDIA hooks, either:

- The adapter selection is silently redirected (Optimus / hybrid-GPU systems),
  causing an adapter mismatch, or
- Internal DXGI state set up by the NVIDIA hooks makes `DuplicateOutput`
  reject the call unconditionally.

Either way the call returns `DXGI_ERROR_UNSUPPORTED` (0x887A0004).

`TestCapture` is unaffected because it does **not** link against any CUDA or
TensorRT libraries. The NVIDIA DLLs are never loaded into that process, so the
DXGI state is pristine when `DuplicateOutput` is called.

## Fix: Delay-Load the CUDA and TensorRT DLLs

The fix uses the MSVC `/DELAYLOAD` linker option to prevent the NVIDIA DLLs
from being loaded at process startup. With delay-loading, the OS loader does
**not** process them at startup; they are loaded on demand the first time any
of their exported functions is actually called at runtime.

In `AutoaimApp`'s lifetime the first CUDA/TensorRT call happens inside
`Detector::init()`, which is invoked **after** `DXGICapture::init()` has
already succeeded. By the time the NVIDIA hooks run, the `IDXGIOutputDuplication`
object is fully constructed and no longer affected.

### CMakeLists.txt changes

```cmake
target_link_libraries(AutoaimApp
    ...
    delayimp          # import library required for /DELAYLOAD support
)

target_link_options(AutoaimApp PRIVATE
    "/DELAYLOAD:cudart64_13.dll"
    "/DELAYLOAD:nvinfer_10.dll"
    "/DELAYLOAD:nvinfer_plugin_10.dll"
    "/DELAYLOAD:nvonnxparser_10.dll"
)
```

`delayimp.lib` provides the `__delayLoadHelper2` stub that the linker inserts
at every call site of a delay-loaded DLL.

### Initialization order (after fix)

```
main()
  └─ Log::Init()
  └─ AutoAimer::init()
        └─ DXGICapture::init()          ← CUDA DLLs NOT yet loaded → succeeds
        └─ Detector::init()             ← first TRT call → cudart/nvinfer loaded here
        └─ Controller::init()
  └─ AutoAimer::run()
```

## Why Other Attempted Fixes Did Not Work

| Approach                                                 | Why it failed                                                                                                                 |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Explicitly enumerating adapters via `IDXGIFactory1`      | The NVIDIA hooks affect `DuplicateOutput` regardless of which adapter/device is used; adapter identity is not the root issue. |
| Using `IDXGIOutput5::DuplicateOutput1`                   | Same hooks apply; the newer API returns the same `DXGI_ERROR_UNSUPPORTED`.                                                    |
| Creating the D3D device on every adapter and trying each | All attempts fail for the same reason — the NVIDIA hooks are process-wide.                                                    |

## References

- [MSDN — IDXGIOutput1::DuplicateOutput](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_2/nf-dxgi1_2-idxgioutput1-duplicateoutput)
- [MSDN — Linker Support for Delay-Loaded DLLs](https://learn.microsoft.com/en-us/cpp/build/reference/linker-support-for-delay-loaded-dlls)
- [CUDA Toolkit Docs — Direct3D 11 Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D11.html)
