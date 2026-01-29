#pragma once
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

class DXGICapture {
public:
	bool init();
	bool capture();

	int width() const { return m_width; }
	int height() const { return m_height; }

private:
	Microsoft::WRL::ComPtr<ID3D11Device> m_device;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_context;
	Microsoft::WRL::ComPtr<IDXGIOutputDuplication> m_duplication;

	Microsoft::WRL::ComPtr<ID3D11Texture2D> m_staging;

	static const int WARM_UP = 3;

	int m_width = 0;
	int m_height = 0;
};