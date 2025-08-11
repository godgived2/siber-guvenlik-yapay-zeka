import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def veri_analizi(dosya_yolu):
    try:
        logging.info(f"Veri dosyası aranıyor: {dosya_yolu}")
        if not os.path.exists(dosya_yolu):
            logging.error(f"Dosya bulunamadı: {dosya_yolu}")
            return
        
        # Veri yükleme
        df = pd.read_csv(dosya_yolu)
        logging.info(f"Veri başarıyla yüklendi. Toplam kayıt sayısı: {len(df)}")
        logging.info(f"Veri sütunları ({len(df.columns)}): {list(df.columns)}")
        logging.info(f"İlk 5 kayıt:\n{df.head()}")
        logging.info(f"Veri şekli: {df.shape}")

        # Eksik değer analizi
        eksik = df.isnull().sum()
        eksik = eksik[eksik > 0]
        if not eksik.empty:
            logging.warning("Eksik değer bulunan sütunlar:")
            logging.warning(f"\n{eksik}")
            # Eksik değer grafiği
            plt.figure(figsize=(10,6))
            eksik.plot(kind='bar', color='orange')
            plt.title("Eksik Değer Sayısı")
            plt.ylabel("Eksik Kayıt Sayısı")
            plt.tight_layout()
            plt.savefig("eksik_degerler.png")
            plt.close()
            logging.info("Eksik değer grafiği 'eksik_degerler.png' olarak kaydedildi.")
        else:
            logging.info("Eksik değer bulunmamaktadır.")

        # Tarih/saat gibi sayısal olmayan sütunlar - çıkar (describe'ta hata olmasın diye)
        sayisal_df = df.select_dtypes(include=[np.number])
        
        # Temel istatistikler sayısal sütunlar için
        logging.info("Sayısal sütunların temel istatistikleri:")
        logging.info(f"\n{sayisal_df.describe()}")

        # Korelasyon matrisi (sayısal sütunlar)
        corr = sayisal_df.corr()
        plt.figure(figsize=(12,10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title("Korelasyon Matrisi")
        plt.tight_layout()
        plt.savefig("korelasyon_matrisi.png")
        plt.close()
        logging.info("Korelasyon matrisi 'korelasyon_matrisi.png' olarak kaydedildi.")

        # Hedef değişken (Label) sınıf dağılımı
        if 'Label' in df.columns:
            label_counts = df['Label'].value_counts()
            toplam = label_counts.sum()
            logging.info("Hedef değişken sınıf dağılımı:")
            for sinif, adet in label_counts.items():
                oran = adet / toplam * 100
                logging.info(f"  Sınıf {sinif}: {adet} örnek (%{oran:.2f})")
            # Dengesizlik uyarısı
            min_oran = (label_counts / toplam).min()
            if min_oran < 0.01:
                logging.warning("Bazı sınıflarda ciddi dengesizlik var. SMOTE gibi teknikler önerilir.")
        else:
            logging.warning("'Label' sütunu veri setinde bulunamadı.")

        logging.info("Veri analizi tamamlandı.")

    except Exception as e:
        logging.error(f"Hata oluştu: {e}")

if __name__ == "__main__":
    veri_dosyasi = r"D:\veria.binetflow"  
    veri_analizi(veri_dosyasi)
