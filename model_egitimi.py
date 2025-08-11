import pandas as pd
import numpy as np
import os
import joblib
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RANDOM_STATE = 42
DATA_PATH = r"D:\veria.binetflow"
MODEL_PATH = "en_iyi_model.pkl"
SAMPLE_SIZE = 100000  # Bellek sorunları için azaltıldı

def veri_yukle_ve_temizle(dosya_yolu):
    if not os.path.exists(dosya_yolu):
        logging.error(f"Veri dosyası bulunamadı: {dosya_yolu}")
        exit(1)
    logging.info(f"Veri yükleniyor: {dosya_yolu}")
    df = pd.read_csv(dosya_yolu)
    logging.info(f"Veri yüklendi, boyut: {df.shape}")

    if "Label" not in df.columns:
        logging.error("'Label' sütunu veri setinde bulunamadı!")
        exit(1)

    df = az_ornekli_siniflari_temizle(df, hedef_sutun="Label", min_ornek=2)
    return df

def az_ornekli_siniflari_temizle(df, hedef_sutun, min_ornek=2):
    counts = df[hedef_sutun].value_counts()
    az_ornekli = counts[counts < min_ornek].index.tolist()
    if az_ornekli:
        logging.info(f"Az örnekli sınıflar çıkarılıyor: {az_ornekli}")
        df = df[~df[hedef_sutun].isin(az_ornekli)]
        logging.info(f"Temizlendikten sonra veri boyutu: {df.shape}")
    return df

def onisleme_ve_encode(X):
    # Tarih/saat sütunlarını otomatik algıla ve kaldır
    tarih_sutunlar = [col for col in X.columns if
                     any(kelime in col.lower() for kelime in ['date', 'time', 'timestamp'])]
    if tarih_sutunlar:
        logging.info(f"Tarih/saat sütunları kaldırılıyor: {tarih_sutunlar}")
        X = X.drop(columns=tarih_sutunlar)

    # Kategorik sütunları LabelEncoder ile sayısala çevir
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Eksik değer varsa median ile doldur
    if X.isnull().sum().sum() > 0:
        logging.info("Eksik değerler bulundu, median ile dolduruluyor (ön işleme)...")
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    return X

def stratified_sample(df, hedef_sutun, ornek_sayisi, random_state):
    X = df.drop(columns=[hedef_sutun])
    y = df[hedef_sutun]

    temp_df = pd.concat([X, y], axis=1)
    temp_df = az_ornekli_siniflari_temizle(temp_df, hedef_sutun, min_ornek=2)
    X = temp_df.drop(columns=[hedef_sutun])
    y = temp_df[hedef_sutun]

    # StratifiedShuffleSplit train_size oranda olmalı, örnek sayısına göre oran hesapla
    oran = ornek_sayisi / len(df)
    if oran > 1.0:
        oran = 1.0

    sss = StratifiedShuffleSplit(n_splits=1, train_size=oran, random_state=random_state)
    for train_idx, _ in sss.split(X, y):
        X_sample = X.iloc[train_idx]
        y_sample = y.iloc[train_idx]

    temp_sample = pd.concat([X_sample, y_sample], axis=1)
    temp_sample = az_ornekli_siniflari_temizle(temp_sample, hedef_sutun, min_ornek=2)
    X_sample = temp_sample.drop(columns=[hedef_sutun])
    y_sample = temp_sample[hedef_sutun]

    logging.info(f"Stratified sampling sonrası veri boyutu: {X_sample.shape}")
    return X_sample, y_sample

def smote_dengesi(X_train, y_train):
    # Eksik değer varsa median ile doldur
    if X_train.isnull().sum().sum() > 0:
        logging.info("SMOTE öncesi eksik değerler median ile dolduruluyor...")
        imputer = SimpleImputer(strategy='median')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

    min_sinif_sayisi = y_train.value_counts().min()
    k_neighbors = min(5, min_sinif_sayisi - 1) if min_sinif_sayisi > 1 else 1
    logging.info(f"SMOTE uygulanıyor, k_neighbors={k_neighbors}...")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logging.info(f"SMOTE sonrası eğitim seti boyutu: {X_res.shape}")
    return X_res, y_res

def modelleri_hazirla():
    return {
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [100],
                "max_depth": [None, 10],
                "min_samples_split": [2, 5]
            }
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }
        ),
        "XGBoost": (
            XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric="mlogloss", use_label_encoder=False),
            {
                "n_estimators": [100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 6]
            }
        ),
        "LightGBM": (
            LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [-1, 10]
            }
        )
    }

def model_egit_ve_sec(X_train, y_train, modeller):
    en_iyi_model = None
    en_iyi_skor = 0
    en_iyi_isim = None

    for isim, (model, param_grid) in modeller.items():
        logging.info(f"{isim} için GridSearchCV başlatılıyor...")
        # Paralel işlemi kapat, bellek sorunu olmaması için n_jobs=1
        grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=1, verbose=1)
        grid.fit(X_train, y_train)

        skor = grid.best_score_
        logging.info(f"{isim} en iyi doğruluk: {skor:.4f}, parametreler: {grid.best_params_}")

        if skor > en_iyi_skor:
            en_iyi_skor = skor
            en_iyi_model = grid.best_estimator_
            en_iyi_isim = isim

    logging.info(f"Seçilen en iyi model: {en_iyi_isim} - CV doğruluk: {en_iyi_skor:.4f}")
    return en_iyi_model, en_iyi_isim

def degerlendir_ve_kaydet(model, X_test, y_test, model_adi):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Test seti doğruluk: {acc:.4f}")
    print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))
    print("\nKarmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    logging.info(f"Model '{MODEL_PATH}' dosyasına kaydedildi.")

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_adi} - Confusion Matrix")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    logging.info("Confusion matrix görselleştirildi ve kaydedildi.")

def main():
    df = veri_yukle_ve_temizle(DATA_PATH)

    X_sample, y_sample = stratified_sample(df, hedef_sutun="Label", ornek_sayisi=SAMPLE_SIZE, random_state=RANDOM_STATE)

    X_sample = onisleme_ve_encode(X_sample)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=RANDOM_STATE
    )
    logging.info(f"Eğitim seti boyutu: {X_train.shape}, Test seti boyutu: {X_test.shape}")

    X_train_res, y_train_res = smote_dengesi(X_train, y_train)

    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)

    en_iyi_model, en_iyi_isim = model_egit_ve_sec(X_train_res, y_train_res, modelleri_hazirla())

    degerlendir_ve_kaydet(en_iyi_model, X_test, y_test, en_iyi_isim)

if __name__ == "__main__":
    main()
