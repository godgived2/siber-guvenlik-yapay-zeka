import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder

#önisleme.py
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def veri_onisleme(dosya_yolu, cikti_dosyasi="hazirlanmis_veri.csv"):
    """
    veria.binetflow veri seti için gelişmiş ön işleme fonksiyonu.
    
    İşlemler:
    - Veri yükleme
    - Eksik değerleri sütun tipine göre doldurma (sayısal: medyan, kategorik: mod)
    - IP adreslerinin 32-bit integer’a dönüştürülmesi (hata varsa NaN)
    - IP adreslerindeki NaN'lar medyan ile doldurulması
    - Zaman sütunundan (StartTime) saat, gün, ay gibi yeni özellikler çıkarılması
    - Kategorik değişkenlerin OrdinalEncoder ile sayısallaştırılması (Label hariç)
    - Özelliklerin StandardScaler ile ölçeklendirilmesi
    - Ön işlenmiş verinin CSV olarak kaydedilmesi
    """

    if not os.path.exists(dosya_yolu):
        logging.error(f"Dosya bulunamadı: {dosya_yolu}")
        return None
    
    logging.info(f"Veri dosyası yükleniyor: {dosya_yolu}")
    df = pd.read_csv(dosya_yolu)

    logging.info(f"Veri başarıyla yüklendi. Kayıt sayısı: {len(df)}, Sütun sayısı: {len(df.columns)}")

    # Eksik değer analizi ve sütun bazında doldurma
    eksik = df.isnull().sum()
    if eksik.any():
        logging.warning(f"Eksik değer bulunan sütunlar:\n{eksik[eksik > 0]}")
        for col in eksik[eksik > 0].index:
            if df[col].dtype in [np.float64, np.int64]:
                medyan = df[col].median()
                df[col] = df[col].fillna(medyan)
                logging.info(f"{col} sütunundaki eksik değerler medyan ({medyan}) ile dolduruldu.")
            else:
                mod = df[col].mode()[0]
                df[col] = df[col].fillna(mod)
                logging.info(f"{col} sütunundaki eksik değerler mod ({mod}) ile dolduruldu.")
    else:
        logging.info("Eksik veri bulunmamaktadır.")

    # IP adreslerini integer'a çevirme fonksiyonu (hata varsa np.nan)
    def ip_to_int(ip):
        try:
            parts = ip.split('.')
            return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
        except Exception:
            return np.nan

    for ip_col in ['SrcAddr', 'DstAddr']:
        if ip_col in df.columns:
            df[ip_col] = df[ip_col].astype(str).apply(ip_to_int)
            # NaN olanları medyan ile doldur
            if df[ip_col].isnull().any():
                medyan_ip = df[ip_col].median()
                df[ip_col] = df[ip_col].fillna(medyan_ip)
                logging.info(f"{ip_col} sütunundaki hatalı IP'ler medyan ({medyan_ip}) ile dolduruldu.")
            logging.info(f"{ip_col} sayısal hale getirildi.")

    # Zaman sütunundan özellik çıkarma (StartTime)
    if 'StartTime' in df.columns:
        logging.info("StartTime sütunu datetime formatına çevriliyor ve yeni zaman özellikleri oluşturuluyor.")
        df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')
        na_count = df['StartTime'].isna().sum()
        if na_count > 0:
            logging.warning(f"StartTime sütununda {na_count} geçersiz datetime değeri bulundu ve silinecek.")
        df.dropna(subset=['StartTime'], inplace=True)
        df['StartHour'] = df['StartTime'].dt.hour
        df['StartDay'] = df['StartTime'].dt.day
        df['StartMonth'] = df['StartTime'].dt.month
        df['StartWeekday'] = df['StartTime'].dt.weekday
        df.drop(columns=['StartTime'], inplace=True)

    # Kategorik değişkenlerin sayısallaştırılması (Label hariç)
    kategorik_sutunlar = df.select_dtypes(include=['object']).columns.tolist()
    if 'Label' in kategorik_sutunlar:
        kategorik_sutunlar.remove('Label')

    if kategorik_sutunlar:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[kategorik_sutunlar] = oe.fit_transform(df[kategorik_sutunlar])
        logging.info(f"Kategorik sütunlar OrdinalEncoder ile sayısal hale getirildi: {kategorik_sutunlar}")
    else:
        logging.info("Sayısallaştırılacak kategorik sütun bulunamadı.")

    # Label sütununu sayısal hale getir
    if 'Label' in df.columns:
        le_label = LabelEncoder()
        df['Label'] = le_label.fit_transform(df['Label'])
        logging.info("Label sütunu sayısal hale getirildi.")
    else:
        le_label = None
        logging.warning("Label sütunu bulunamadı!")

    # Özellikler ve hedef değişkeni ayır
    if 'Label' in df.columns:
        X = df.drop(columns=['Label'])
        y = df['Label']
    else:
        X = df.copy()
        y = None

    # Özelliklerin ölçeklendirilmesi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    logging.info("Özellikler StandardScaler ile ölçeklendirildi.")

    # Hedef değişkenle birleştir
    if y is not None:
        df_onislenmis = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)
    else:
        df_onislenmis = X_scaled_df

    # CSV olarak kaydet
    df_onislenmis.to_csv(cikti_dosyasi, index=False)
    logging.info(f"Ön işlenmiş veri '{cikti_dosyasi}' olarak kaydedildi.")

    return df_onislenmis, oe if kategorik_sutunlar else None, scaler, le_label


if __name__ == "__main__":
    dosya = r"D:\veria.binetflow"
    veri_onisleme(dosya)
