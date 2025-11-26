import numpy as np
from PIL import Image
import math

class SimpleJPEG:
    def __init__(self, quality=50):
        self.quality = quality
        # Простая таблица квантования
        self.quant_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])
        
        # Настройка качества
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality
        self.quant_table = np.maximum(1, (self.quant_table * scale) / 100)

    def dct(self, block):
        """Простое DCT преобразование блока 8x8"""
        result = np.zeros((8, 8))
        for u in range(8):
            for v in range(8):
                sum_val = 0
                for i in range(8):
                    for j in range(8):
                        cos1 = math.cos((2*i+1) * u * math.pi / 16)
                        cos2 = math.cos((2*j+1) * v * math.pi / 16)
                        sum_val += block[i,j] * cos1 * cos2
                cu = 1/math.sqrt(2) if u == 0 else 1
                cv = 1/math.sqrt(2) if v == 0 else 1
                result[u,v] = 0.25 * cu * cv * sum_val
        return result

    def idct(self, block):
        """Обратное DCT преобразование"""
        result = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                sum_val = 0
                for u in range(8):
                    for v in range(8):
                        cu = 1/math.sqrt(2) if u == 0 else 1
                        cv = 1/math.sqrt(2) if v == 0 else 1
                        cos1 = math.cos((2*i+1) * u * math.pi / 16)
                        cos2 = math.cos((2*j+1) * v * math.pi / 16)
                        sum_val += cu * cv * block[u,v] * cos1 * cos2
                result[i,j] = 0.25 * sum_val
        return result

    def compress_image(self, image_path):
        """Сжатие изображения"""
        # Загрузка изображения
        img = Image.open(image_path).convert('L')  # В оттенки серого
        img_array = np.array(img)
        
        height, width = img_array.shape
        compressed_blocks = []
        
        # Обработка блоков 8x8
        for y in range(0, height-7, 8):
            for x in range(0, width-7, 8):
                block = img_array[y:y+8, x:x+8].astype(float) - 128
                
                # DCT
                dct_block = self.dct(block)
                
                # Квантование
                quantized = np.round(dct_block / self.quant_table)
                compressed_blocks.append(quantized)
        
        return compressed_blocks, (height, width)

    def decompress_image(self, compressed_blocks, original_size):
        """Восстановление изображения"""
        height, width = original_size
        reconstructed = np.zeros((height, width))
        block_idx = 0
        
        for y in range(0, height-7, 8):
            for x in range(0, width-7, 8):
                quantized = compressed_blocks[block_idx]
                
                # Обратное квантование
                dct_block = quantized * self.quant_table
                
                # Обратное DCT
                block = self.idct(dct_block) + 128
                
                reconstructed[y:y+8, x:x+8] = block
                block_idx += 1
        
        return np.clip(reconstructed, 0, 255).astype(np.uint8)

def main():
    # Создаем простое тестовое изображение
    print("Создание тестового изображения...")
    img_array = np.zeros((64, 64), dtype=np.uint8)
    
    # Рисуем простые фигуры
    for i in range(64):
        for j in range(64):
            if 20 <= i < 40 and 20 <= j < 40:
                img_array[i,j] = 255  # Белый квадрат
            elif (i-32)**2 + (j-32)**2 < 100:
                img_array[i,j] = 128  # Серый круг
            else:
                img_array[i,j] = 50   # Темный фон
    
    # Сохраняем оригинал
    original_img = Image.fromarray(img_array)
    original_img.save('original.png')
    
    print("Тестирование сжатия JPEG...")
    
    for quality in [10, 25, 50, 75]:
        print(f"\nКачество: {quality}%")
        
        jpeg = SimpleJPEG(quality)
        
        # Сохраняем и загружаем чтобы имитировать файл
        temp_img = Image.fromarray(img_array)
        temp_img.save('temp_input.png')
        
        # Сжимаем
        compressed, size = jpeg.compress_image('temp_input.png')
        
        # Восстанавливаем
        reconstructed = jpeg.decompress_image(compressed, size)
        
        # Сохраняем результат
        result_img = Image.fromarray(reconstructed)
        result_img.save(f'compressed_q{quality}.png')
        
        # Вычисляем качество
        mse = np.mean((img_array.astype(float) - reconstructed.astype(float))**2)
        psnr = 20 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else float('inf')
        
        print(f"PSNR: {psnr:.2f} dB")
        print(f"MSE: {mse:.2f}")
    
    print("\nГотово! Проверьте файлы:")
    print("original.png - оригинальное изображение")
    print("compressed_qXX.png - сжатые версии")

if __name__ == "__main__":
    main()
