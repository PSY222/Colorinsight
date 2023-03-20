import colour
import numpy as np

# RGB 값 (0~255)을 [0, 1] 범위로 정규화합니다.
rgb = np.array([255, 0, 0]) / 255.0  # 예시로 빨간색(RGB: 255, 0, 0)을 사용

# sRGB 색상 공간에서 RGB를 Munsell로 변환합니다.
munsell = colour.notation.RGB_to_Munsell(rgb, illuminant='D65', method='McCamy')

# Munsell 색상 체계로 변환된 색상을 출력합니다.
print(munsell)  # MunsellSpecification(hue=..., value=..., chroma=...)