from PIL import Image, ExifTags
import numpy as np
from io import BytesIO
import pandas as pd
import streamlit as st

class ImageProcessor:
    def __init__(self, image, a, b, Le, s):
        self.image = image
        self.a = a
        self.b = b
        self.Le = Le
        self.s = s
        self.processed_image = None
        self.processed_luminance = None  # 加工後の輝度を保持
        self.overflown_luminance = None  # クリッピング前の輝度を保持
        self.max_luminance = None
        self.width = None
        self.height = None

    def process_image(self):
        """
        画像をグレースケールに変換し、補正式を適用して加工します。
        """
        # 画像を NumPy 配列に変換し、0-1 に正規化
        img_array = np.array(self.image).astype(np.float32) / 255.0

        # 画像の幅と高さを取得
        self.height, self.width = img_array.shape[:2]

        # RGB から輝度 Y を計算（グレースケール変換）
        Y = 0.2126 * img_array[:, :, 0] + \
            0.7152 * img_array[:, :, 1] + \
            0.0722 * img_array[:, :, 2]

        # 補正式を適用（クリッピング前の輝度を保持）
        self.overflown_luminance = (self.Le / (self.s * self.b)) * \
                                   (np.exp(Y / self.a) - 1)

        # 補正後の最大輝度を計算
        self.max_luminance = np.max(self.overflown_luminance)

        # 画像として保存する場合はクリッピングするが、元データはそのまま保持
        self.processed_luminance = np.clip(self.overflown_luminance, 0.0, 1.0)

        # グレースケール画像を NumPy 配列に変換し、輝度値を更新（クリッピング後の値）
        processed_img = self.processed_luminance * 255.0
        self.processed_image = processed_img.astype(np.uint8)

    def get_processed_image(self):
        """
        クリッピングされた加工済みの画像を取得します。
        """
        return self.processed_image

    def get_processed_image_csv(self):
        """
        処理後の画像のピクセル値をクリッピングせずにCSVファイルとして取得します。
        """
        # 画像の幅と高さを取得（process_imageで取得済み）
        width = self.width
        height = self.height

        # クリッピング前のピクセル値を取得
        processed_img_array = self.overflown_luminance

        # R, G, B の値を取得してリストに変換
        r_values = processed_img_array.flatten()
        g_values = processed_img_array.flatten()
        b_values = processed_img_array.flatten()

        # DataFrame を作成
        df = pd.DataFrame({
            'R': r_values,
            'G': g_values,
            'B': b_values
        })

        # 最初に width と height を書くために、テキストバッファを使用
        csv_buffer = BytesIO()
        csv_buffer.write(f"{width},{height}\n".encode('utf-8'))

        # ピクセル値を CSV に追記
        df.to_csv(csv_buffer, index=False, header=False)

        csv_buffer.seek(0)
        return csv_buffer

# Streamlit アプリケーション部分
def main():
    st.title('実機撮影画像を理想カメラで撮影した画像に加工しましょう')

    # 画像のアップロード
    uploaded_file = st.file_uploader('JPG 画像を選択してください', type=['jpg', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='アップロードされた画像', use_column_width=True)

        # EXIFから露光時間を取得
        exif_data = image._getexif()
        exposure_time = None
        if exif_data:
            exif = {
                ExifTags.TAGS.get(key): value
                for key, value in exif_data.items()
                if key in ExifTags.TAGS
            }
            exposure_time = exif.get('ExposureTime', None)
            if exposure_time and isinstance(exposure_time, tuple):
                exposure_time = exposure_time[0] / exposure_time[1]

        # 露光時間の初期値を設定（EXIFデータがない場合は 1.0）
        s_initial = float(exposure_time) if exposure_time else 1.0

        # パラメータ入力フォームの追加
        st.write('### パラメータの入力')
        a = st.number_input('パラメータ a', min_value=1e-6, value=0.34430231806854655, format="%.6f")
        b = st.number_input('パラメータ b', min_value=1e-6, value=5936.953535493492, format="%.6f")
        Le = st.number_input('パラメータ Le', min_value=1e-6, value=50.0, format="%.6f")
        s = st.slider('露光時間 (s)', min_value=0.1, max_value=10.0, value=s_initial, step=0.1)

        # ImageProcessor インスタンスを作成
        processor = ImageProcessor(image, a, b, Le, s)

        # グレースケール画像を生成して表示
        processor.process_image()
        gray_image = processor.get_processed_image()
        st.image(gray_image, caption='グレースケールおよび補正後の画像', use_column_width=True)

        # 補正後の最大輝度を表示
        st.write(f'補正後の最大輝度: {processor.max_luminance}')

        # 補正式の説明
        st.write('### 補正式')
        st.write('以下の補正式を適用して画像を加工します。')
        st.latex(r'Y = 0.2126 \cdot R + 0.7152 \cdot G + 0.0722 \cdot B')
        st.latex(rf'L = \frac{{{processor.Le}}}{{{processor.s} \cdot {processor.b}}} \cdot \left( e^{{Y/{processor.a}}} - 1 \right)')
        st.latex(r'R, G, B: \text{RGB 色空間の各チャンネル}')
        st.latex(r'Y: \text{輝度}')
        st.latex(r'L: \text{補正後の輝度}')
        st.latex(r'a, b: \text{パラメータ}')
        st.latex(r'L_e: \text{最大輝度}')
        st.latex(r's: \text{露光時間}')

        # 画像のダウンロードボタンの追加
        img_buffer = BytesIO()
        result_image = Image.fromarray(gray_image)
        result_image.save(img_buffer, format="JPEG")
        img_buffer.seek(0)
        st.download_button(
            label="処理後の画像をダウンロード",
            data=img_buffer,
            file_name="processed_image.jpg",
            mime="image/jpeg"
        )

        # ピクセル値を CSV としてダウンロード（クリッピングなし）
        csv_buffer = processor.get_processed_image_csv()
        st.download_button(
            label="処理後の画像のピクセル値を CSV でダウンロード",
            data=csv_buffer,
            file_name="image_pixels.csv",
            mime="text/csv"
        )
    else:
        st.write('画像をアップロードしてください。')

if __name__ == '__main__':
    main()
