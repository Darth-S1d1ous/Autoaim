#include "dxgi_capture.h"
#include <log/Log.h>
#include <iostream>

bool DXGICapture::init() {
	HRESULT hr;

	// Create D3D device
	hr = D3D11CreateDevice(
		nullptr,
		D3D_DRIVER_TYPE_HARDWARE,
		nullptr,
		0,
		nullptr,
		0,
		D3D11_SDK_VERSION,
		&m_device,
		nullptr,
		&m_context
	);
	if (FAILED(hr)) {
		CORE_ERROR("Create device failed!");
		return false;
	}

	// Get DXGI device
	Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
	m_device.As(&dxgiDevice);

	Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
	dxgiDevice->GetAdapter(&adapter);

	// Get output (primary monitor)
	Microsoft::WRL::ComPtr<IDXGIOutput> output;
	adapter->EnumOutputs(0, &output);

	DXGI_OUTPUT_DESC desc;
	output->GetDesc(&desc);

	m_width = desc.DesktopCoordinates.right - desc.DesktopCoordinates.left;
	m_height = desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top;

	// Duplicate output
	Microsoft::WRL::ComPtr<IDXGIOutput1> output1;
	output.As(&output1);

	hr = output1->DuplicateOutput(m_device.Get(), &m_duplication);
	if (FAILED(hr)) {
		CORE_ERROR("Duplicate output failed!");
		return false;
	}

	// create staging texture (CPU readable)
	D3D11_TEXTURE2D_DESC texDesc = {};
	texDesc.Width = m_width;
	texDesc.Height = m_height;
	texDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	texDesc.ArraySize = 1;
	texDesc.MipLevels = 1;
	texDesc.SampleDesc.Count = 1;
	texDesc.Usage = D3D11_USAGE_STAGING;
	texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

	hr = m_device->CreateTexture2D(&texDesc, nullptr, &m_staging);
	if (FAILED(hr)) {
		CORE_ERROR("Create texture failed!");
		return false;
	}

	CORE_INFO("DXGI Capture init OK");
	
	return true;
}

bool DXGICapture::capture() {
	DXGI_OUTDUPL_FRAME_INFO frameInfo;
	Microsoft::WRL::ComPtr<IDXGIResource> resource;

	HRESULT hr = m_duplication->AcquireNextFrame(
		16, // timeout ms
		&frameInfo,
		&resource
	);
	if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
		return false;
	}
	if (FAILED(hr)) {
		CORE_ERROR("AcquireNextFrame failed!");
		return false;
	}

	Microsoft::WRL::ComPtr<ID3D11Texture2D> frame;
	resource.As(&frame);

	// Copy to cpu texture
	m_context->CopyResource(m_staging.Get(), frame.Get());

	D3D11_MAPPED_SUBRESOURCE mapped;
	hr = m_context->Map(m_staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
	if (SUCCEEDED(hr)) {
		// mapped.pData is BGRA buffer
		// mapped.RowPitch is stride
		// 👉 这里后面直接转 cv::Mat / TensorRT
		m_context->Unmap(m_staging.Get(), 0);
	}
	else {
		CORE_ERROR("Copy texture failed!");
	}

	m_duplication->ReleaseFrame();
	return true;
}