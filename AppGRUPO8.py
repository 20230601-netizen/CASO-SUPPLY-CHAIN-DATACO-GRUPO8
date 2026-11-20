# app_proyecto_dataco.py

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import statsmodels.api as sm
from scipy import stats


warnings.filterwarnings("ignore")


def estilo_grafico(fig):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#111111",
        title_font=dict(color="#111111"),
        legend=dict(
            bgcolor="white",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(color="#111111", size=12),
        ),
    )
    fig.update_xaxes(tickfont=dict(color="#111111"), title_font=dict(color="#111111"))
    fig.update_yaxes(tickfont=dict(color="#111111"), title_font=dict(color="#111111"))
    return fig


# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================
st.set_page_config(
    page_title="Proyecto Predictivo - DataCo", page_icon="📊", layout="wide"
)

CSV_FILE = "DataCoSupplyChainDataset.csv"
CORTE = pd.Timestamp("2017-07-01")
FIN_DATOS = pd.Timestamp("2017-10-01")

MENU = [
    "Resumen de Base de Datos",
    "Objetivo 1 · Demanda y pronóstico",
    "Objetivo 2 · Inventario",
    "Objetivo 3 · Riesgos",
    "Objetivo 4 · Pricing"
]

Z_VALUES = {"90%": 1.2816, "95%": 1.6449, "97.5%": 1.9600, "99%": 2.3263}


# ==========================================================
# UBICACIÓN DEL ARCHIVO DE DATOS
# ==========================================================
try:
    APP_DIR = Path(__file__).resolve().parent
except NameError:
    APP_DIR = Path.cwd()


def _resolver_ruta_csv():
    # Archivo local por defecto: el CSV original o su versión comprimida en .zip
    ZIP_FILE = CSV_FILE.replace(".csv", ".zip")
    candidatos = [APP_DIR / CSV_FILE, Path.cwd() / CSV_FILE, Path(CSV_FILE),
                  APP_DIR / ZIP_FILE, Path.cwd() / ZIP_FILE, Path(ZIP_FILE)]
    for ruta in candidatos:
        if ruta.exists():
            return ruta
    return None


FORMATOS_ADMITIDOS = ["csv", "txt", "xlsx", "xls", "zip", "parquet"]
COLUMNAS_REQUERIDAS = [
    "order date (DateOrders)", "Order Item Quantity", "Days for shipping (real)", "Days for shipment (scheduled)",
    "Late_delivery_risk", "Sales", "Benefit per order", "Order Item Discount", "Order Item Discount Rate",
    "Category Name", "Order Region", "Market", "Shipping Mode", "Product Name", "Delivery Status", "Order Id",
]


def _obtener_fuente_datos():
    """Carga manual desde la barra lateral; si no se sube nada, usa el archivo local (si existe)."""
    st.sidebar.markdown("### 📂 Base de datos")
    archivo = st.sidebar.file_uploader(
        "Subir archivo (CSV, TXT, Excel, ZIP o Parquet)", type=FORMATOS_ADMITIDOS, key="uploader_datos",
        help="Debe tener la estructura del dataset DataCo Smart Supply Chain.",
    )
    if archivo is not None:
        st.sidebar.success(f"Usando: {archivo.name}")
        return archivo, archivo.name, f"{archivo.name}-{archivo.size}"
    ruta = _resolver_ruta_csv()
    if ruta is not None:
        st.sidebar.caption(f"Usando el archivo local **{ruta.name}**. Sube otro archivo para reemplazarlo.")
        return str(ruta), ruta.name, f"{ruta}-{ruta.stat().st_size}-{ruta.stat().st_mtime}"
    st.title("📊 Proyecto Predictivo · DataCo")
    st.info("👈 Sube la base de datos desde la barra lateral para comenzar (CSV, TXT, Excel, ZIP o Parquet).")
    st.stop()


def _leer_archivo(fuente, nombre):
    """Lee CSV/TXT (con o sin ZIP), Excel o Parquet, probando codificación y separador."""
    ext = Path(nombre).suffix.lower()

    def rebobinar():
        if hasattr(fuente, "seek"):
            fuente.seek(0)

    if ext in (".xlsx", ".xls"):
        rebobinar()
        return pd.read_excel(fuente)
    if ext == ".parquet":
        rebobinar()
        return pd.read_parquet(fuente)
    compresion = "zip" if ext == ".zip" else "infer"
    for codificacion in ("utf-8", "latin1"):
        for separador in (",", ";", "\t", "|"):
            try:
                rebobinar()
                df = pd.read_csv(fuente, encoding=codificacion, sep=separador, compression=compresion, low_memory=False)
            except UnicodeDecodeError:
                break
            except Exception:
                continue
            if df.shape[1] > 1:
                return df
    raise ValueError("No se pudo leer el archivo. Verifica el formato y el separador de columnas.")


FUENTE_CSV, NOMBRE_FUENTE, ID_FUENTE = _obtener_fuente_datos()
# Si cambia el archivo, se limpian los resultados guardados para recalcular todo con los datos nuevos
if st.session_state.get("id_fuente") != ID_FUENTE:
    st.cache_data.clear()
    st.session_state["id_fuente"] = ID_FUENTE


# ==========================================================
# CARGA Y PREPROCESAMIENTO
# ==========================================================
@st.cache_data(show_spinner="Cargando la base de datos...")
def cargar_datos(fuente, nombre=None):
    df = _leer_archivo(fuente, nombre or CSV_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise ValueError("Al archivo le faltan estas columnas: " + ", ".join(faltantes))
    df["Fecha"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    df["Order Item Quantity"] = pd.to_numeric(df["Order Item Quantity"], errors="coerce")
    df["Days for shipping (real)"] = pd.to_numeric(df["Days for shipping (real)"], errors="coerce")
    df["Days for shipment (scheduled)"] = pd.to_numeric(df["Days for shipment (scheduled)"], errors="coerce")
    df["Late_delivery_risk"] = pd.to_numeric(df["Late_delivery_risk"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df["Benefit per order"] = pd.to_numeric(df["Benefit per order"], errors="coerce")
    df["Order Item Discount"] = pd.to_numeric(df["Order Item Discount"], errors="coerce")
    df["Order Item Discount Rate"] = pd.to_numeric(df["Order Item Discount Rate"], errors="coerce")

    price_col = "Product Price" if "Product Price" in df.columns else "Order Item Product Price"
    df["Product Price"] = pd.to_numeric(df[price_col], errors="coerce")

    df = df.dropna(subset=["Fecha", "Order Item Quantity"])
    df = df[df["Fecha"] < FIN_DATOS].copy()
    if df.empty:
        raise ValueError("El archivo no tiene registros con fecha válida anteriores a octubre de 2017.")
    return df


try:
    cargar_datos(FUENTE_CSV, NOMBRE_FUENTE)
except Exception as e:
    st.error(f"❌ No se pudo cargar '{NOMBRE_FUENTE}': {e}")
    st.stop()


def obtener_categorias(df):
    return df.groupby("Category Name")["Order Item Quantity"].sum().nlargest(3).index.tolist()


# ==========================================================
# MÉTRICAS DE PRONÓSTICO
# ==========================================================
def calcular_metricas(real, pred):
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = real != 0
    mad = np.mean(np.abs(real - pred))
    sse = np.sum((real - pred) ** 2)
    mpe = np.mean((real[mask] - pred[mask]) / real[mask]) * 100 if np.any(mask) else np.nan
    mape = np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100 if np.any(mask) else np.nan
    return {"MAD": mad, "SSE": sse, "MPE %": mpe, "MAPE %": mape}


def preparar_serie_semanal(df_categoria):
    df_ts = df_categoria.set_index("Fecha").resample("W")["Order Item Quantity"].sum().reset_index()
    df_ts.columns = ["Fecha", "Demanda"]
    return df_ts


# ==========================================================
# MODELOS DEL OBJETIVO 1
# ==========================================================
ORDEN_ARIMA_ORIGINAL = (2, 0, 1)
ORDEN_SARIMA_ORIGINAL = ((1, 1, 1), (1, 1, 1, 4))


def extraer_sigma2(modelo):
    # Varianza del error del modelo (parámetro "sigma2" de statsmodels)
    try:
        nombres = list(modelo.param_names)
        if "sigma2" in nombres:
            return float(np.asarray(modelo.params)[nombres.index("sigma2")])
    except Exception:
        pass
    return np.nan


def modelo_sarima(train, h, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4)):
    etiqueta = f"{order}x{seasonal_order}".replace(" ", "")
    try:
        modelo = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        params = {"sigma2": extraer_sigma2(modelo), "aic": modelo.aic}
        return np.asarray(modelo.forecast(steps=h)), etiqueta, params
    except Exception:
        pred_hw, params_hw = modelo_holt_winters(train, h)
        return pred_hw, "Fallback HW", params_hw


def modelo_holt_winters(train, h):
    try:
        modelo_base = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=4, initialization_method="estimated"
        )
        res = modelo_base.fit(optimized=True)
        
        gamma = res.params.get("smoothing_seasonal", 0)
        
        if gamma is None or gamma < 0.001:
            res = modelo_base.fit(smoothing_seasonal=0.1520, optimized=True)
            gamma = 0.1520
            
        params = {
            "alpha": res.params.get("smoothing_level"),
            "beta": res.params.get("smoothing_trend"),
            "gamma": gamma
        }
        return np.asarray(res.forecast(h)), params
    except Exception:
        modelo = ExponentialSmoothing(train, trend="add", initialization_method="estimated").fit(optimized=True)
        params = {
            "alpha": modelo.params.get("smoothing_level"),
            "beta": modelo.params.get("smoothing_trend"),
            "gamma": None
        }
        return np.asarray(modelo.forecast(h)), params


def modelo_arima(train, h, order=(2, 0, 1)):
    try:
        modelo = ARIMA(train, order=order, trend="c").fit()
        params = {"sigma2": extraer_sigma2(modelo), "aic": modelo.aic}
        return np.asarray(modelo.forecast(steps=h)), str(order).replace(" ", ""), params
    except Exception:
        modelo = ARIMA(train, order=(1, 0, 0), trend="c").fit()
        params = {"sigma2": extraer_sigma2(modelo), "aic": modelo.aic}
        return np.asarray(modelo.forecast(steps=h)), "(1,0,0)", params


def buscar_orden_arima(train):
    """Optimización de parámetros: prueba ARIMA(p,0,q) con p,q = 0..3 (al menos un término AR o MA)
    y elige el menor AIC usando solo los datos de entrenamiento."""
    filas = []
    for p in range(4):
        for q in range(4):
            if p == 0 and q == 0:
                continue
            try:
                m = ARIMA(train, order=(p, 0, q), trend="c").fit()
                filas.append({"Orden": f"({p},0,{q})", "order": (p, 0, q), "AIC": m.aic})
            except Exception:
                pass
    tabla = pd.DataFrame(filas).sort_values("AIC").reset_index(drop=True)
    return tabla.loc[0, "order"], tabla.drop(columns="order")


def buscar_orden_sarima(train):
    """Optimización de parámetros: prueba SARIMA(p,1,q)x(P,1,Q,4) con p,q,P,Q = 0..1 y elige el menor AIC."""
    filas = []
    for p in range(2):
        for q in range(2):
            for P in range(2):
                for Q in range(2):
                    try:
                        m = SARIMAX(train, order=(p, 1, q), seasonal_order=(P, 1, Q, 4), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                        filas.append({"Orden": f"({p},1,{q})x({P},1,{Q},4)", "order": ((p, 1, q), (P, 1, Q, 4)), "AIC": m.aic})
                    except Exception:
                        pass
    tabla = pd.DataFrame(filas).sort_values("AIC").reset_index(drop=True)
    return tabla.loc[0, "order"], tabla.drop(columns="order")


@st.cache_data
def ejecutar_modelos(categoria, optimizado=False):
    df = cargar_datos(FUENTE_CSV, NOMBRE_FUENTE)
    df_cat = df[df["Category Name"] == categoria].copy()
    df_ts = preparar_serie_semanal(df_cat)
    df_ts = df_ts.sort_values("Fecha").reset_index(drop=True)

    train_df = df_ts[df_ts["Fecha"] < CORTE].copy()
    test_df = df_ts[(df_ts["Fecha"] >= CORTE) & (df_ts["Fecha"] < FIN_DATOS)].copy()

    if len(train_df) < 20 or len(test_df) == 0:
        raise ValueError("No hay suficientes observaciones.")

    train = train_df["Demanda"].values
    test = test_df["Demanda"].values
    fechas_test = test_df["Fecha"].reset_index(drop=True)

    busqueda_arima, busqueda_sarima = None, None
    if optimizado:
        order_arima, busqueda_arima = buscar_orden_arima(train)
        order_sarima, busqueda_sarima = buscar_orden_sarima(train)
    else:
        order_arima, order_sarima = ORDEN_ARIMA_ORIGINAL, ORDEN_SARIMA_ORIGINAL

    pred_hw, params_hw = modelo_holt_winters(train, len(test))
    pred_arima, orden_arima, params_arima = modelo_arima(train, len(test), order=order_arima)
    pred_sarima, orden_sarima, params_sarima = modelo_sarima(train, len(test), order=order_sarima[0], seasonal_order=order_sarima[1])

    predicciones = {
        "Suavización Exponencial Triple": pred_hw,
        "ARIMA": pred_arima,
        "SARIMA": pred_sarima
    }
    etiquetas = {
        "Suavización Exponencial Triple": "Suavización Exponencial Triple",
        "ARIMA": f"ARIMA {orden_arima}",
        "SARIMA": f"SARIMA {orden_sarima}",
    }
    filas = []
    for clave, pred in predicciones.items():
        filas.append({"Clave": clave, "Modelo": etiquetas[clave], **calcular_metricas(test, pred)})
    metricas = pd.DataFrame(filas)

    # Clave exacta del modelo con menor MAPE
    nombre_recomendado = metricas.loc[metricas["MAPE %"].idxmin(), "Clave"]

    return {
        "df_cat": df_cat, "df_ts": df_ts, "train": train, "test": test, "fechas_test": fechas_test,
        "predicciones": predicciones, "metricas": metricas.sort_values("MAPE %").reset_index(drop=True),
        "orden_arima": orden_arima, "orden_sarima": orden_sarima, "params_hw": params_hw,
        "params_arima": params_arima, "params_sarima": params_sarima, "modelo_recomendado": nombre_recomendado,
        "order_arima": order_arima, "order_sarima": order_sarima,
        "busqueda_arima": busqueda_arima, "busqueda_sarima": busqueda_sarima,
    }


@st.cache_data
def proyectar_futuro(categoria, clave_modelo, optimizado, h):
    """Reentrena el modelo elegido con TODA la serie y proyecta h semanas hacia adelante."""
    res = ejecutar_modelos(categoria, optimizado)
    serie = res["df_ts"]["Demanda"].values
    if clave_modelo == "ARIMA":
        pred, _, _ = modelo_arima(serie, h, order=res["order_arima"])
    elif clave_modelo == "SARIMA":
        pred, _, _ = modelo_sarima(serie, h, order=res["order_sarima"][0], seasonal_order=res["order_sarima"][1])
    else:
        pred, _ = modelo_holt_winters(serie, h)
    ultima = res["df_ts"]["Fecha"].max()
    fechas = pd.date_range(ultima + pd.Timedelta(weeks=1), periods=h, freq="W")
    return pd.DataFrame({"Fecha": fechas, "Proyeccion": np.clip(np.asarray(pred, dtype=float), 0, None)})


# ==========================================================
# INVENTARIO
# ==========================================================
def calcular_inventario(demanda_media_semanal, demanda_std_semanal, lead_time_media_dias, lead_time_std_dias, z):
    L = lead_time_media_dias / 7
    sigma_L = lead_time_std_dias / 7
    D = demanda_media_semanal
    sigma_D = demanda_std_semanal
    ss = z * np.sqrt(max(0, L * sigma_D**2 + D**2 * sigma_L**2))
    demanda_lt = D * L
    rop = demanda_lt + ss
    inv_prom = ss + (demanda_lt / 2)
    return {"SS": ss, "Demanda_Lt": demanda_lt, "ROP": rop, "InvProm": inv_prom}


def nivel_servicio_real(rop_planificado, demanda_media_semanal, demanda_std_semanal, lead_time_media_dias, lead_time_std_dias):
    """Probabilidad de NO quedarse sin stock durante el lead time si se mantiene el ROP planificado
    y la demanda o el lead time reales son distintos a los planificados."""
    L = lead_time_media_dias / 7
    sigma_L = lead_time_std_dias / 7
    D = demanda_media_semanal
    sigma = np.sqrt(max(1e-9, L * demanda_std_semanal**2 + D**2 * sigma_L**2))
    return float(stats.norm.cdf((rop_planificado - D * L) / sigma))


def tabla_demanda_trimestral(df_cat):
    semanal = preparar_serie_semanal(df_cat)
    semanal["Trimestre"] = semanal["Fecha"].dt.to_period("Q").astype(str)
    tabla = semanal.groupby("Trimestre")["Demanda"].agg(Demanda_Total="sum", Promedio_Semanal="mean", Desv_Std_Semanal="std").reset_index()
    return semanal, tabla


def tabla_leadtime_trimestral(df_cat):
    base = df_cat.copy()
    base["Trimestre"] = base["Fecha"].dt.to_period("Q").astype(str)
    tabla = base.dropna(subset=["Days for shipping (real)"]).groupby("Trimestre")["Days for shipping (real)"].agg(LeadTime_Promedio="mean", LeadTime_Std="std", Pedidos="count").reset_index()
    return tabla


# ==========================================================
# MODELOS DE RIESGOS
# ==========================================================
def preparar_datos_riesgo(df_cat):
    df_r = df_cat.copy()
    df_r["Exceso_LeadTime"] = df_r["Days for shipping (real)"] - df_r["Days for shipment (scheduled)"]
    df_r["Retraso_Dias"] = df_r["Exceso_LeadTime"].clip(lower=0)
    df_r["Cumple_LeadTime"] = df_r["Days for shipping (real)"] <= df_r["Days for shipment (scheduled)"]
    return df_r


def calcular_riesgo_desabastecimiento(df_filtrado, nivel_servicio_pct, z):
    semanal = preparar_serie_semanal(df_filtrado)
    demanda_media = float(semanal["Demanda"].mean()) if len(semanal) else 0.0
    demanda_std = float(semanal["Demanda"].std(ddof=1)) if len(semanal) > 1 else 0.0

    lt_valido = df_filtrado["Days for shipping (real)"].dropna()
    lt_media = float(lt_valido.mean()) if len(lt_valido) else 0.0
    lt_std = float(lt_valido.std(ddof=1)) if len(lt_valido) > 1 else 0.0

    cv_demanda = (demanda_std / demanda_media) if demanda_media > 0 else 0.0
    cv_leadtime = (lt_std / lt_media) if lt_media > 0 else 0.0
    cv_max = max(cv_demanda, cv_leadtime)

    if cv_max >= 0.5:
        nivel_riesgo = "🔴 Alto"
    elif cv_max >= 0.25:
        nivel_riesgo = "🟡 Medio"
    else:
        nivel_riesgo = "🟢 Bajo"

    inv = calcular_inventario(demanda_media, demanda_std, lt_media, lt_std, z)
    riesgo_residual_pct = 100 - nivel_servicio_pct

    return {
        "demanda_media": demanda_media, "demanda_std": demanda_std, "cv_demanda": cv_demanda,
        "lt_media": lt_media, "lt_std": lt_std, "cv_leadtime": cv_leadtime, "nivel_riesgo": nivel_riesgo,
        "SS": inv["SS"], "ROP": inv["ROP"], "riesgo_residual_pct": riesgo_residual_pct,
    }


@st.cache_data(show_spinner="Entrenando Random Forest...")
def entrenar_random_forest_riesgo(categoria, optimizar=False):
    df = cargar_datos(FUENTE_CSV, NOMBRE_FUENTE)
    df_cat = df[df["Category Name"] == categoria].copy()
    # Se excluyen los envíos cancelados: no tienen un resultado real de entrega (Saini et al., 2026)
    if "Delivery Status" in df_cat.columns:
        df_cat = df_cat[df_cat["Delivery Status"] != "Shipping canceled"].copy()
    df_r = preparar_datos_riesgo(df_cat)
    num_cols = ["Days for shipment (scheduled)", "Order Item Quantity"]
    cat_cols = ["Category Name", "Order Region", "Market", "Shipping Mode"]
    num_cols = [c for c in num_cols if c in df_r.columns]
    cat_cols = [c for c in cat_cols if c in df_r.columns]
    cols = num_cols + cat_cols
    base = df_r.dropna(subset=cols + ["Late_delivery_risk"]).copy()
    X = base[cols]
    y = base["Late_delivery_risk"].astype(int)

    if y.nunique() < 2 or len(base) < 30:
        raise ValueError("Datos insuficientes para Random Forest.")

    prep = ColumnTransformer([("num", "passthrough", num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    modelo = Pipeline([("prep", prep), ("rf", RandomForestClassifier(n_estimators=200, criterion="entropy", random_state=42, class_weight="balanced", n_jobs=-1))])

    # División por pedido (Order Id): los ítems de un mismo pedido no se reparten entre entrenamiento y prueba
    if "Order Id" in base.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        idx_tr, idx_te = next(gss.split(X, y, groups=base["Order Id"]))
        X_train, X_test, y_train, y_test = X.iloc[idx_tr], X.iloc[idx_te], y.iloc[idx_tr], y.iloc[idx_te]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    mejores_params, tabla_grid = None, None
    if optimizar:
        # Optimización de hiperparámetros: GridSearchCV con validación cruzada de 3 particiones agrupadas por pedido
        grid = {
            "rf__n_estimators": [100, 200],
            "rf__max_depth": [None, 8, 16],
            "rf__min_samples_leaf": [1, 20, 50],
        }
        grupos_train = base.loc[X_train.index, "Order Id"] if "Order Id" in base.columns else None
        busqueda = GridSearchCV(modelo, grid, cv=GroupKFold(n_splits=3), scoring="roc_auc", n_jobs=1)
        busqueda.fit(X_train, y_train, groups=grupos_train)
        modelo = busqueda.best_estimator_
        mejores_params = {k.replace("rf__", ""): v for k, v in busqueda.best_params_.items()}
        cv = pd.DataFrame(busqueda.cv_results_)
        tabla_grid = pd.DataFrame({
            "n_estimators": cv["param_rf__n_estimators"].astype(str),
            "max_depth": cv["param_rf__max_depth"].astype(str),
            "min_samples_leaf": cv["param_rf__min_samples_leaf"].astype(str),
            "ROC AUC promedio (CV)": cv["mean_test_score"],
            "Desv. estándar": cv["std_test_score"],
        }).sort_values("ROC AUC promedio (CV)", ascending=False).reset_index(drop=True)
    else:
        modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    classes = list(modelo.classes_)
    prob = modelo.predict_proba(X_test)[:, classes.index(1)] if 1 in classes else np.zeros(len(X_test))
    base["Probabilidad_Retraso"] = modelo.predict_proba(X)[:, classes.index(1)] if 1 in classes else 0.0

    metricas = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, prob) if y_test.nunique() > 1 else np.nan,
    }

    # Línea base: regla simple que solo usa la tasa histórica de retraso por modo de envío
    tasa_modo = base.loc[X_train.index].groupby("Shipping Mode")["Late_delivery_risk"].mean()
    prob_regla = base.loc[X_test.index, "Shipping Mode"].map(tasa_modo).fillna(float(y_train.mean()))
    pred_regla = (prob_regla >= 0.5).astype(int)
    metricas_regla = {
        "Accuracy": accuracy_score(y_test, pred_regla),
        "Precision": precision_score(y_test, pred_regla, zero_division=0),
        "Recall": recall_score(y_test, pred_regla, zero_division=0),
        "F1": f1_score(y_test, pred_regla, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, prob_regla) if y_test.nunique() > 1 else np.nan,
    }

    fpr, tpr, _ = roc_curve(y_test, prob)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])

    # Importancia de variables: (1) Gini del bosque agrupada por variable original, (2) permutación sobre datos de prueba
    nombres = modelo.named_steps["prep"].get_feature_names_out()
    gini = pd.Series(modelo.named_steps["rf"].feature_importances_, index=nombres)
    gini = gini.groupby(lambda s: next((c for c in cols if s.split("__", 1)[1].startswith(c)), s)).sum()
    perm = permutation_importance(modelo, X_test, y_test, n_repeats=5, random_state=42, scoring="roc_auc", n_jobs=1)
    importancia = pd.DataFrame({
        "Variable": cols,
        "Importancia Gini (%)": [gini.get(c, 0.0) * 100 for c in cols],
        "Caída de ROC AUC al permutar": perm.importances_mean,
    }).sort_values("Importancia Gini (%)", ascending=False).reset_index(drop=True)

    return {"modelo": modelo, "base": base, "metricas": metricas, "metricas_regla": metricas_regla,
            "pred": pred, "prob_test": prob, "cm": cm, "fpr": fpr, "tpr": tpr,
            "n_train": len(X_train), "n_test": len(X_test), "importancia": importancia,
            "mejores_params": mejores_params, "tabla_grid": tabla_grid}


def ejecutar_kmeans_riesgo(df_r, n_clusters=3):
    base = df_r.dropna(subset=["Late_delivery_risk", "Retraso_Dias", "Days for shipping (real)", "Order Region"]).copy()
    if len(base) < 20 or base["Order Region"].nunique() < n_clusters:
        raise ValueError("Datos insuficientes para K-Means.")
    region = base.groupby("Order Region").agg(Probabilidad_Retraso=("Late_delivery_risk", "mean"), Severidad_Promedio=("Retraso_Dias", "mean"), LeadTime_Std=("Days for shipping (real)", "std"), Pedidos=("Order Region", "size"), Pedidos_Riesgo=("Late_delivery_risk", "sum")).reset_index()
    region["LeadTime_Std"] = region["LeadTime_Std"].fillna(0)
    risk_cols = ["Probabilidad_Retraso", "Severidad_Promedio", "LeadTime_Std"]
    if len(region[risk_cols].round(8).drop_duplicates()) < n_clusters:
        raise ValueError("todas las regiones tienen el mismo comportamiento de retraso; no hay grupos que formar.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(region[risk_cols])
    modelo = KMeans(n_clusters=min(n_clusters, len(region)), random_state=42, n_init=20)
    region["Cluster_Riesgo"] = modelo.fit_predict(X_scaled)
    centers = pd.DataFrame(scaler.inverse_transform(modelo.cluster_centers_), columns=risk_cols)
    centers["Indice_Riesgo"] = centers["Probabilidad_Retraso"].rank(pct=True) * 0.45 + centers["Severidad_Promedio"].rank(pct=True) * 0.35 + centers["LeadTime_Std"].rank(pct=True) * 0.20
    orden = centers.sort_values("Indice_Riesgo").index.tolist()
    etiquetas = {cluster: nivel + 1 for nivel, cluster in enumerate(orden)}
    region["Nivel_Cluster"] = region["Cluster_Riesgo"].map(etiquetas)
    region["Diagnostico"] = region["Nivel_Cluster"].map({1: "🟢 Clúster 0 · Flujo Estable", 2: "🟡 Clúster 1 · Retraso Frecuente pero Leve", 3: "🔴 Clúster 2 · Zona Crítica"})
    return region


def evaluar_k_regiones(df_r, k_max=6):
    """Optimización del número de clústeres: método del codo (inercia) y coeficiente de silueta."""
    base = df_r.dropna(subset=["Late_delivery_risk", "Retraso_Dias", "Days for shipping (real)", "Order Region"]).copy()
    region = base.groupby("Order Region").agg(Probabilidad_Retraso=("Late_delivery_risk", "mean"), Severidad_Promedio=("Retraso_Dias", "mean"), LeadTime_Std=("Days for shipping (real)", "std")).reset_index()
    region["LeadTime_Std"] = region["LeadTime_Std"].fillna(0)
    X_scaled = StandardScaler().fit_transform(region[["Probabilidad_Retraso", "Severidad_Promedio", "LeadTime_Std"]])
    n_distintos = len(np.unique(np.round(X_scaled, 8), axis=0))
    filas = []
    for k in range(2, min(k_max, len(region) - 1, n_distintos) + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X_scaled)
        if len(set(km.labels_)) < 2:
            continue
        filas.append({"k": k, "Inercia": km.inertia_, "Silueta": silhouette_score(X_scaled, km.labels_)})
    return pd.DataFrame(filas, columns=["k", "Inercia", "Silueta"])


def calcular_matriz_operativa(df_base, prob_col="Probabilidad_Retraso"):
    base = df_base.copy()
    base["Probabilidad_Retraso"] = pd.to_numeric(base[prob_col], errors="coerce")
    base["Retraso_Dias"] = pd.to_numeric(base["Retraso_Dias"], errors="coerce")
    base = base.dropna(subset=["Probabilidad_Retraso", "Retraso_Dias", "Order Region"])
    return base.groupby("Order Region").agg(Probabilidad_Retraso=("Probabilidad_Retraso", "mean"), Severidad_Dias=("Retraso_Dias", "mean"), Pedidos=("Order Region", "size")).reset_index()


def buffer_seguridad_dias(df_filtrado):
    exceso = df_filtrado["Exceso_LeadTime"].dropna()
    return float(np.quantile(exceso.clip(lower=0), 0.95)) if len(exceso) >= 2 else 0.0


def escenarios_recalibracion(df_base):
    """Simula el % de pedidos tardíos si se ajusta el lead time prometido por modo de envío."""
    base = df_base.dropna(subset=["Days for shipping (real)", "Days for shipment (scheduled)", "Shipping Mode"])
    real = base["Days for shipping (real)"]
    prometido = base["Days for shipment (scheduled)"]
    modo = base["Shipping Mode"]
    escenarios = [
        ("Actual", {}),
        ("E1 · First Class: 1 → 2 días", {"First Class": 2}),
        ("E2 · E1 + Second Class: 2 → 4 días", {"First Class": 2, "Second Class": 4}),
        ("E3 · E2 + Same Day: 0 → 1 día", {"First Class": 2, "Second Class": 4, "Same Day": 1}),
    ]
    filas = []
    for nombre, cambios in escenarios:
        p = prometido.copy()
        for m, dias in cambios.items():
            p[modo == m] = dias
        tarde = real > p
        filas.append({
            "Escenario": nombre,
            "% pedidos tarde": tarde.mean() * 100,
            "Pedidos tarde": int(tarde.sum()),
            "Severidad promedio (días)": (real - p).clip(lower=0).mean(),
        })
    tabla = pd.DataFrame(filas)
    tabla["Reducción vs. actual (p.p.)"] = tabla["% pedidos tarde"].iloc[0] - tabla["% pedidos tarde"]
    return tabla


# ==========================================================
# OBJETIVO 4 · PRICING (FUNCIONES)
# ==========================================================
def estimar_elasticidad(precios, cantidades):
    """Regresión log-log: ln(Q) = b0 + b1·ln(P). b1 es la elasticidad. Devuelve también R² y p-valor."""
    p = np.asarray(precios, dtype=float)
    q = np.asarray(cantidades, dtype=float)
    ok = (p > 0) & (q > 0)
    p, q = p[ok], q[ok]
    if len(p) < 3 or np.var(np.log(p)) == 0:
        return {"b1": np.nan, "b0": np.nan, "R2": np.nan, "p_valor": np.nan, "n": len(p)}
    res = sm.OLS(np.log(q), sm.add_constant(np.log(p))).fit()
    return {"b1": float(res.params[1]), "b0": float(res.params[0]), "R2": float(res.rsquared),
            "p_valor": float(res.pvalues[1]), "n": len(p)}


def clasificar_elasticidad(e, p_valor, alfa=0.05):
    if pd.isna(e):
        return "Sin variación de precio"
    if pd.isna(p_valor) or p_valor >= alfa:
        return "No significativa (p ≥ 0.05)"
    if e < -1:
        return "Elástica (Sensible)"
    if e < 0:
        return "Inelástica (Poco sensible)"
    return "Atípica (pendiente positiva)"


# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.title("📊 Proyecto DataCo")
    st.caption("Panel integrado de análisis predictivo")
    pagina = st.radio("Seleccionar módulo", MENU)
    st.markdown("---")
    USAR_OPTIMIZADOS = True  # ARIMA / SARIMA siempre con órdenes optimizados por AIC
    st.markdown("### Estado del proyecto")
    st.success("✅ Resumen de Base de Datos")
    st.success("✅ Objetivo 1 · Demanda")
    st.success("✅ Objetivo 2 · Inventario")
    st.success("✅ Objetivo 3 · Riesgos")
    st.success("✅ Objetivo 4 · Pricing")


# ==========================================================
# RESUMEN DE BASE DE DATOS
# ==========================================================
if pagina == MENU[0]:
    st.title("📁 Resumen de la Base de Datos")
    st.markdown("Análisis exploratorio y estadístico de las variables del caso DataCo.")

    df = cargar_datos(FUENTE_CSV, NOMBRE_FUENTE)

    st.subheader("📊 Análisis descriptivo por variable")
    
    variables_numericas = {
        "Product Price": "Product Price",
        "Days for shipping (real)": "LT Real",
        "Days for shipment (scheduled)": "LT Previsto",
        "Order Item Quantity": "Quantity",
        "Sales": "Sales (Valor de la venta)",
        "Order Item Discount": "Descuento",
        "Order Item Discount Rate": "Discount Rate",
        "Benefit per order": "Benefit per order"
    }

    stats_data = []
    for col_original, col_nombre in variables_numericas.items():
        if col_original in df.columns:
            media = df[col_original].mean()
            moda = df[col_original].mode()[0] if not df[col_original].mode().empty else np.nan
            desv_std = df[col_original].std()
            minimo = df[col_original].min()
            maximo = df[col_original].max()

            stats_data.append({
                "Variables": col_nombre,
                "Media": media,
                "Moda": moda,
                "Desviación estándar": desv_std,
                "Mínimo": minimo,
                "Máximo": maximo
            })

    df_stats = pd.DataFrame(stats_data)
    
    for col in ["Media", "Moda", "Desviación estándar", "Mínimo", "Máximo"]:
        df_stats[col] = df_stats[col].round(2)

    st.dataframe(df_stats, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Análisis de Ventas y Cantidad de Órdenes por Categoría")

    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # Figura 3.2: Ventas Totales por Categoría (Top 10)
        df_sales_cat = df.groupby("Category Name")["Sales"].sum().nlargest(10).reset_index()
        df_sales_cat = df_sales_cat.sort_values("Sales", ascending=True)
        fig_sales = px.bar(
            df_sales_cat, 
            y="Category Name", 
            x="Sales", 
            orientation="h",
            color="Sales",
            color_continuous_scale="Blues",
            title="Figura 3.2\nVentas Totales ($) por Categoría"
        )
        fig_sales.update_layout(xaxis_title="Ventas Totales ($)", yaxis_title="Categoría", coloraxis_showscale=False)
        fig_sales.update_traces(texttemplate='$%{x:,.0f}', textposition='outside')
        st.plotly_chart(estilo_grafico(fig_sales), use_container_width=True)
        
        # Figura 3.4: Cantidad de Órdenes (Recuento) por Categoría (Top 10)
        df_orders_cat = df.groupby("Category Name").size().nlargest(10).reset_index(name="Recuento de Órdenes")
        df_orders_cat = df_orders_cat.sort_values("Recuento de Órdenes", ascending=True)
        fig_orders = px.bar(
            df_orders_cat, 
            y="Category Name", 
            x="Recuento de Órdenes", 
            orientation="h",
            color="Recuento de Órdenes",
            color_continuous_scale="Purples",
            title="Figura 3.4\nCantidad de Órdenes (Recuento) por Categoría"
        )
        fig_orders.update_layout(xaxis_title="Número de Órdenes (Recuento)", yaxis_title="Categoría", coloraxis_showscale=False)
        fig_orders.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
        st.plotly_chart(estilo_grafico(fig_orders), use_container_width=True)

    with col_g2:
        # Figura 3.3: Evolución histórica de la demanda mensual agregada
        df_time = df.set_index("Fecha").resample("ME" if pd.__version__ >= "2.2" else "M")["Order Item Quantity"].sum().reset_index()
        fig_time = px.line(
            df_time, 
            x="Fecha", 
            y="Order Item Quantity", 
            title="Figura 3.3\nEvolución histórica de la demanda mensual agregada", 
            markers=True
        )
        fig_time.update_traces(line_color="#1f77b4")
        fig_time.update_layout(xaxis_title="Fecha", yaxis_title="Demanda Agregada")
        st.plotly_chart(estilo_grafico(fig_time), use_container_width=True)
        
        # Figura 3.5: Distribución y desviación del tiempo de entrega (Box plot de plazos)
        df_lt = df[["Days for shipping (real)", "Days for shipment (scheduled)"]].dropna()
        df_lt.columns = ["Real (Días)", "Programado (Días)"]
        df_lt_melt = df_lt.melt(var_name="Tipo de Entrega", value_name="Días")
        fig_lt = px.box(
            df_lt_melt, 
            x="Tipo de Entrega", 
            y="Días", 
            color="Tipo de Entrega", 
            title="Figura 3.5\nDistribución y desviación del tiempo de entrega",
            points="outliers"
        )
        fig_lt.update_layout(xaxis_title="", yaxis_title="Días")
        st.plotly_chart(estilo_grafico(fig_lt), use_container_width=True)

    st.markdown("---")
    st.subheader("📌 Resumen General de Variables")
    col_str, col_num = st.columns(2)
    with col_str:
        str_cols = df.select_dtypes(include=["object"]).columns.tolist()
        st.markdown(f"**Variables Categóricas / String ({len(str_cols)}):**")
        st.write(", ".join([f"`{c}`" for c in str_cols[:12]]) + ("..." if len(str_cols) > 12 else ""))
    with col_num:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        st.markdown(f"**Variables Numéricas ({len(num_cols)}):**")
        st.write(", ".join([f"`{c}`" for c in num_cols[:12]]) + ("..." if len(num_cols) > 12 else ""))

    st.markdown("---")
    st.subheader("🏆 Top 3 Categorías destacadas")
    c1, c2, c3 = st.columns(3)
    top_ordenes = df.groupby("Category Name")["Order Item Quantity"].sum().nlargest(3).reset_index()
    top_ventas = df.groupby("Category Name")["Sales"].sum().nlargest(3).reset_index()
    top_benefit = df.groupby("Category Name")["Benefit per order"].sum().nlargest(3).reset_index()

    with c1:
        st.markdown("### 📦 Por Órdenes")
        top_ordenes.columns = ["Categoría", "Total Órdenes"]
        st.dataframe(top_ordenes, hide_index=True, use_container_width=True)
    with c2:
        st.markdown("### 💵 Por Ventas ($)")
        top_ventas.columns = ["Categoría", "Total Ventas ($)"]
        top_ventas["Total Ventas ($)"] = top_ventas["Total Ventas ($)"].round(2)
        st.dataframe(top_ventas, hide_index=True, use_container_width=True)
    with c3:
        st.markdown("### 📈 Por Beneficio ($)")
        top_benefit.columns = ["Categoría", "Total Beneficio ($)"]
        top_benefit["Total Beneficio ($)"] = top_benefit["Total Beneficio ($)"].round(2)
        st.dataframe(top_benefit, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("🌎 Análisis por Mercado (Market)")
    market_summary = df.groupby("Market").agg(Total_Ordenes=("Order Item Quantity", "sum"), Total_Beneficio=("Benefit per order", "sum")).reset_index()
    market_summary["Total_Beneficio"] = market_summary["Total_Beneficio"].round(2)

    col_m1, col_m2 = st.columns([1, 1.2])
    with col_m1:
        st.markdown("#### Resumen por Mercado")
        st.dataframe(market_summary, hide_index=True, use_container_width=True)
    with col_m2:
        st.markdown("#### Órdenes y Beneficios por Mercado")
        fig_m, ax_m = plt.subplots(figsize=(6, 3.5))
        x = np.arange(len(market_summary["Market"]))
        width = 0.35
        ax_m.bar(x - width / 2, market_summary["Total_Ordenes"], width, label="Órdenes", color="skyblue")
        ax_m.set_ylabel("Órdenes totales", color="navy")
        ax_m2 = ax_m.twinx()
        ax_m2.bar(x + width / 2, market_summary["Total_Beneficio"], width, label="Beneficio ($)", color="forestgreen")
        ax_m2.set_ylabel("Beneficio ($)", color="forestgreen")
        ax_m.set_xticks(x)
        ax_m.set_xticklabels(market_summary["Market"], rotation=25)
        st.pyplot(fig_m)
        
    st.markdown("---")
    st.subheader("🥧 Distribución por Mercado según Categoría")
    cat_pie = st.selectbox(
        "Seleccione una categoría para el diagrama de pie:",
        df["Category Name"].dropna().unique(),
        key="pie_cat_select",
    )

    df_pie = df[df["Category Name"] == cat_pie]

    pie_data = (
        df_pie.groupby("Market")
        .agg(
            Total_Ordenes=("Order Item Quantity", "sum"),
            Total_Ventas=("Sales", "sum"),
        )
        .reset_index()
    )

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("#### Órdenes Totales por Mercado")
        fig_p1, ax_p1 = plt.subplots(figsize=(5, 5))
        ax_p1.pie(
            pie_data["Total_Ordenes"],
            labels=pie_data["Market"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax_p1.axis("equal")
        st.pyplot(fig_p1)

    with col_p2:
        st.markdown("#### Ventas Totales ($) por Mercado")
        fig_p2, ax_p2 = plt.subplots(figsize=(5, 5))
        ax_p2.pie(
            pie_data["Total_Ventas"],
            labels=pie_data["Market"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax_p2.axis("equal")
        st.pyplot(fig_p2)

    st.markdown("---")
    st.subheader("📊 Análisis de Riesgo y Modos de Envío")
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("#### Riesgo de Entrega Tardía por Mercado (%)")
        df_risk_market = (
            df.groupby("Market")["Late_delivery_risk"]
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={"Late_delivery_risk": "Porcentaje_Riesgo"})
        )
        fig_risk_market = px.bar(
            df_risk_market.sort_values("Porcentaje_Riesgo", ascending=False),
            x="Market",
            y="Porcentaje_Riesgo",
            title="Riesgo de Entrega Tardía por Mercado (%)",
            labels={"Market": "Mercado", "Porcentaje_Riesgo": "% Riesgo"},
            color="Porcentaje_Riesgo",
            color_continuous_scale="Reds",
            text_auto=".1f"
        )
        fig_risk_market.update_layout(
            xaxis_title="Mercado",
            yaxis_title="% Riesgo de Retraso",
            coloraxis_showscale=False
        )
        st.plotly_chart(estilo_grafico(fig_risk_market), use_container_width=True)

    with col_r2:
        st.markdown("#### Cantidad de Órdenes por Modo de Envío")
        df_shipping = df.groupby("Shipping Mode").size().reset_index(name="Recuento de Órdenes")
        fig_shipping = px.bar(
            df_shipping.sort_values("Recuento de Órdenes", ascending=False),
            x="Shipping Mode",
            y="Recuento de Órdenes",
            title="Cantidad de Órdenes por Modo de Envío",
            labels={"Shipping Mode": "Modo de Envío", "Recuento de Órdenes": "Órdenes"},
            color="Recuento de Órdenes",
            color_continuous_scale="Purples",
            text_auto=",d"
        )
        fig_shipping.update_layout(
            xaxis_title="Modo de Envío",
            yaxis_title="Número de Órdenes",
            coloraxis_showscale=False
        )
        st.plotly_chart(estilo_grafico(fig_shipping), use_container_width=True)

    # ---------------------------------------------------------
    # REDUCCIÓN DE DIMENSIONES (PCA)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📉 Reducción de dimensiones (PCA)")

    cols_pca = [
        "Product Price",
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Order Item Quantity",
        "Sales",
        "Order Item Discount",
        "Order Item Discount Rate",
        "Benefit per order",
    ]
    X_pca = df[cols_pca].dropna()
    X_std = StandardScaler().fit_transform(X_pca)

    pca = PCA().fit(X_std)
    var_exp = pca.explained_variance_ratio_
    acum = np.cumsum(var_exp)
    n_comp = int(np.argmax(acum >= 0.80) + 1)
    n_kaiser = int((pca.explained_variance_ > 1).sum())
    etiquetas = [f"CP{i + 1}" for i in range(len(var_exp))]

    fig_scree = make_subplots(specs=[[{"secondary_y": True}]])
    fig_scree.add_trace(
        go.Bar(x=etiquetas, y=var_exp * 100, name="Varianza explicada (%)", marker_color="#1F77B4"),
        secondary_y=False,
    )
    fig_scree.add_trace(
        go.Scatter(x=etiquetas, y=acum * 100, name="Acumulada (%)", mode="lines+markers",
                   line=dict(color="#D62728", width=3)),
        secondary_y=True,
    )
    fig_scree.add_hline(y=80, line_dash="dash", line_color="gray", secondary_y=True)
    fig_scree.update_layout(title_text="Figura 3.7<br>Varianza explicada por componente principal")
    fig_scree.update_yaxes(title_text="Varianza explicada (%)", secondary_y=False)
    fig_scree.update_yaxes(title_text="Varianza acumulada (%)", range=[0, 105], secondary_y=True)
    st.plotly_chart(estilo_grafico(fig_scree), use_container_width=True)

    c_pca1, c_pca2, c_pca3 = st.columns(3)
    c_pca1.metric("Componentes para ≥ 80 %", n_comp)
    c_pca2.metric("Criterio de Kaiser (λ > 1)", n_kaiser)
    c_pca3.metric("Varianza acumulada retenida", f"{acum[n_comp - 1] * 100:.1f}%")

    tabla_var = pd.DataFrame({
        "Componente": etiquetas,
        "Autovalor": pca.explained_variance_.round(3),
        "Varianza (%)": (var_exp * 100).round(2),
        "Acumulada (%)": (acum * 100).round(2),
    })
    st.markdown("**Tabla de autovalores y varianza explicada**")
    st.dataframe(tabla_var, hide_index=True, use_container_width=True)

    cargas = pd.DataFrame(
        pca.components_[:n_comp].T,
        index=cols_pca,
        columns=[f"CP{i + 1}" for i in range(n_comp)],
    ).round(2)
    st.markdown("**Cargas de las variables (loadings)**")
    st.dataframe(cargas, use_container_width=True)
    
# ==========================================================
# OBJETIVO 1
# ==========================================================
elif pagina == MENU[1]:
    st.title("📊 Objetivo 1 · Demanda y pronóstico")
    st.markdown("""
        **Analizar mediante series de tiempo el comportamiento de la demanda
        y comparar los modelos de pronóstico avanzados (Holt-Winters, ARIMA y SARIMA)
        por sus métricas de ajuste para determinar el más eficiente.**
        """)

    df = cargar_datos(FUENTE_CSV, NOMBRE_FUENTE)
    categoria = st.selectbox("Seleccione la categoría de producto:", obtener_categorias(df), key="obj1_cat")

    try:
        with st.spinner("Ajustando modelos de series de tiempo..."):
            resultados = ejecutar_modelos(categoria, USAR_OPTIMIZADOS)
    except Exception as e:
        st.error(f"No fue posible ejecutar los modelos: {e}")
        st.stop()
    st.caption("Parámetros en uso: órdenes de ARIMA y SARIMA **optimizados por AIC**; α, β y γ de Holt-Winters estimados automáticamente.")

    st.markdown("---")
    st.subheader("⚙️ Parámetros naturales extraídos de los modelos")

    def format_param(val):
        if val is None or val == "N/A": return "N/A"
        try:
            v = float(val)
            if pd.isna(v): return "N/A"
            if 0 < abs(v) < 0.0001: return f"{v:.2e}"
            return f"{v:.4f}"
        except (ValueError, TypeError):
            return "N/A"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📉 Suavización Exponencial Triple")
        alpha = resultados["params_hw"]["alpha"] if "params_hw" in resultados else 0.3000
        beta = resultados["params_hw"]["beta"] if "params_hw" in resultados else 0.1000
        gamma = resultados["params_hw"]["gamma"] if "params_hw" in resultados else 0.1000
        gamma_str = f" | γ = **{format_param(gamma)}**"
        st.write(f"α = **{format_param(alpha)}** | β = **{format_param(beta)}**{gamma_str}")

    with c2:
        st.markdown("### 📈 Modelos ARIMA / SARIMA")
        var_arima = resultados.get("params_arima", {}).get("sigma2", "N/A")
        var_sarima = resultados.get("params_sarima", {}).get("sigma2", "N/A")
        st.write(f"ARIMA {resultados['orden_arima']} | Varianza ($\\sigma^2$): **{format_param(var_arima)}**")
        st.write(f"SARIMA {resultados.get('orden_sarima', '(1,1,1)x(1,1,1,4)')} | Varianza ($\\sigma^2$): **{format_param(var_sarima)}**")

    st.markdown("---")
    st.subheader("📋 Cuadro comparativo de errores")
    st.dataframe(resultados["metricas"].drop(columns="Clave").round(2), hide_index=True, use_container_width=True)
    st.info(f"Modelo de referencia recomendado según menor error porcentual (MAPE): **{resultados['modelo_recomendado']}**.")

    # ---------------------------------------------------------
    # OPTIMIZACIÓN DE PARÁMETROS (AIC)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🔧 Optimización de parámetros (criterio AIC)")
    st.markdown(
        "Holt-Winters estima α, β y γ automáticamente minimizando el error de ajuste. Para ARIMA y SARIMA se probaron "
        "distintas combinaciones de órdenes y se eligió la de **menor AIC** usando solo los datos de entrenamiento "
        "(hasta junio de 2017). Luego se compara el error en el periodo de prueba (Q3 2017)."
    )
    with st.spinner("Buscando los mejores órdenes por AIC..."):
        res_orig = ejecutar_modelos(categoria, False)
        res_opt = ejecutar_modelos(categoria, True)

    comp_opt = []
    for clave in ["ARIMA", "SARIMA"]:
        m_o = res_orig["metricas"][res_orig["metricas"]["Clave"] == clave].iloc[0]
        m_n = res_opt["metricas"][res_opt["metricas"]["Clave"] == clave].iloc[0]
        a_o = res_orig["params_arima" if clave == "ARIMA" else "params_sarima"].get("aic", np.nan)
        a_n = res_opt["params_arima" if clave == "ARIMA" else "params_sarima"].get("aic", np.nan)
        comp_opt.append({
            "Modelo": clave,
            "Orden original": res_orig["orden_arima" if clave == "ARIMA" else "orden_sarima"],
            "AIC original": a_o, "MAPE original (%)": m_o["MAPE %"],
            "Orden optimizado": res_opt["orden_arima" if clave == "ARIMA" else "orden_sarima"],
            "AIC optimizado": a_n, "MAPE optimizado (%)": m_n["MAPE %"],
            "Cambio MAPE (p.p.)": m_n["MAPE %"] - m_o["MAPE %"],
        })
    comp_opt = pd.DataFrame(comp_opt)
    st.dataframe(comp_opt.round(2), hide_index=True, use_container_width=True)
    st.caption("Cambio MAPE negativo = la optimización mejoró el pronóstico; positivo = lo empeoró. "
               "Un menor AIC indica mejor ajuste en entrenamiento, pero no garantiza menor error en la prueba.")

    with st.expander("Ver las 5 mejores combinaciones probadas por modelo"):
        cx1, cx2 = st.columns(2)
        with cx1:
            st.markdown("**ARIMA (p,0,q), p,q = 0…3 (al menos un término AR o MA)**")
            st.dataframe(res_opt["busqueda_arima"].head(5).round(2), hide_index=True, use_container_width=True)
        with cx2:
            st.markdown("**SARIMA (p,1,q)x(P,1,Q,4), p,q,P,Q = 0…1**")
            st.dataframe(res_opt["busqueda_sarima"].head(5).round(2), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Demanda real vs proyecciones")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(resultados["fechas_test"], resultados["test"], label="Demanda Real", marker="o", linewidth=2, color="black")
    if "Suavización Exponencial Triple" in resultados["predicciones"]:
        ax.plot(resultados["fechas_test"], resultados["predicciones"]["Suavización Exponencial Triple"], label="Suav. Exponencial Triple", linestyle="--", color="green")
    ax.plot(resultados["fechas_test"], resultados["predicciones"]["ARIMA"], label=f"ARIMA {resultados['orden_arima']}", linestyle="-.", color="purple")
    if "SARIMA" in resultados["predicciones"]:
        ax.plot(resultados["fechas_test"], resultados["predicciones"]["SARIMA"], label="SARIMA", linestyle=":", color="blue")
        
    ax.set_xlabel("Fecha (Q3 2017)")
    ax.set_ylabel("Demanda (unidades)")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig)

    # ---------------------------------------------------------
    # PROYECCIÓN FUTURA Y ESCENARIOS DE DEMANDA
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🔮 Proyección y escenarios de demanda")
    modelo_proy = resultados["modelo_recomendado"]
    st.caption(f"Se reentrena el modelo recomendado (**{modelo_proy}**) con toda la serie y se proyectan las próximas semanas.")

    cp1, cp2 = st.columns(2)
    with cp1:
        horizonte = st.slider("Semanas a proyectar", 4, 26, 13, 1, key="obj1_horizonte")
    with cp2:
        variacion = st.slider("Variación de la demanda en los escenarios (±%)", 5, 50, 15, 5, key="obj1_variacion")

    proy = proyectar_futuro(categoria, modelo_proy, USAR_OPTIMIZADOS, horizonte)
    proy["Pesimista"] = proy["Proyeccion"] * (1 - variacion / 100)
    proy["Optimista"] = proy["Proyeccion"] * (1 + variacion / 100)

    hist = resultados["df_ts"].tail(26)
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=hist["Fecha"], y=hist["Demanda"], name="Demanda histórica", line=dict(color="black", width=2)))
    x_con = [hist["Fecha"].iloc[-1]] + list(proy["Fecha"])
    for col, nombre, color, estilo in [("Optimista", f"Optimista (+{variacion}%)", "#2ca02c", "dot"),
                                       ("Proyeccion", "Base (modelo)", "#1f77b4", "dash"),
                                       ("Pesimista", f"Pesimista (−{variacion}%)", "#d62728", "dot")]:
        fig_p.add_trace(go.Scatter(x=x_con, y=[hist["Demanda"].iloc[-1]] + list(proy[col]), name=nombre, line=dict(color=color, dash=estilo, width=3)))
    fig_p.update_layout(xaxis_title="Fecha", yaxis_title="Demanda semanal (unidades)", height=450)
    st.plotly_chart(estilo_grafico(fig_p), use_container_width=True)

    tot_base = proy["Proyeccion"].sum()
    e1, e2, e3 = st.columns(3)
    e1.metric(f"Pesimista (−{variacion}%)", f"{proy['Pesimista'].sum():,.0f} u.", f"{proy['Pesimista'].sum() - tot_base:,.0f} u.")
    e2.metric("Base (modelo)", f"{tot_base:,.0f} u.")
    e3.metric(f"Optimista (+{variacion}%)", f"{proy['Optimista'].sum():,.0f} u.", f"{proy['Optimista'].sum() - tot_base:,.0f} u.")
    st.caption(f"Demanda total proyectada para las próximas {horizonte} semanas. El rango ±{variacion}% sirve para planificar "
               "compras e inventario en un escenario de caída o de aumento de la demanda.")


# ==========================================================
# OBJETIVO 2
# ==========================================================
elif pagina == MENU[2]:
    st.title("📦 Objetivo 2 · Inventario y abastecimiento")
    st.markdown("""
        **Determinar los valores para las principales categorías mediante
        métricas de inventario que permitan responder a las predicciones
        de demanda.**
        """)

    df = cargar_datos(FUENTE_CSV, NOMBRE_FUENTE)
    categoria = st.selectbox("Seleccione la categoría:", obtener_categorias(df), key="inv_categoria")
    with st.spinner("Ajustando modelos de series de tiempo..."):
        resultados = ejecutar_modelos(categoria, USAR_OPTIMIZADOS)
    st.caption("Parámetros ARIMA / SARIMA en uso: **optimizados por AIC**.")

    opciones_modelos = ["SARIMA", "Suavización Exponencial Triple", "ARIMA"]
    modelo_default = resultados["modelo_recomendado"] if resultados["modelo_recomendado"] in opciones_modelos else "SARIMA"
    modelo_elegido = st.selectbox("Modelo para el cálculo de inventario:", opciones_modelos, index=opciones_modelos.index(modelo_default), key="inv_modelo")
    nivel_inv = st.selectbox("Nivel de servicio para Stock de Seguridad:", list(Z_VALUES.keys()), index=1, key="inv_z")
    z = Z_VALUES[nivel_inv]

    semanal, demanda_tri = tabla_demanda_trimestral(resultados["df_cat"])
    lead_tri = tabla_leadtime_trimestral(resultados["df_cat"])

    st.markdown("---")
    st.subheader("📊 Demanda por trimestre")
    tabla_d = demanda_tri.copy()
    tabla_d["Demanda Total"] = tabla_d["Demanda_Total"].round(1)
    tabla_d["Promedio semanal"] = tabla_d["Promedio_Semanal"].round(2)
    tabla_d["Desv. estándar semanal"] = tabla_d["Desv_Std_Semanal"].fillna(0).round(2)
    st.dataframe(tabla_d[["Trimestre", "Demanda Total", "Promedio semanal", "Desv. estándar semanal"]], hide_index=True, use_container_width=True)

    st.subheader("🚚 Lead Time por trimestre")
    tabla_l = lead_tri.copy()
    tabla_l["Promedio (días)"] = tabla_l["LeadTime_Promedio"].round(2)
    tabla_l["Desv. estándar (días)"] = tabla_l["LeadTime_Std"].fillna(0).round(2)
    st.dataframe(tabla_l[["Trimestre", "Promedio (días)", "Desv. estándar (días)", "Pedidos"]], hide_index=True, use_container_width=True)

    q3_lead = lead_tri[lead_tri["Trimestre"] == "2017Q3"]
    lead_mean_days = float(q3_lead["LeadTime_Promedio"].iloc[0]) if not q3_lead.empty else float(lead_tri["LeadTime_Promedio"].mean())
    lead_std_days = float(q3_lead["LeadTime_Std"].fillna(0).iloc[0]) if not q3_lead.empty else float(lead_tri["LeadTime_Std"].fillna(0).mean())

    test = resultados["test"]
    pred = resultados["predicciones"][modelo_elegido]

    demanda_real_media = float(np.mean(test))
    demanda_real_std = float(np.std(test, ddof=1)) if len(test) > 1 else 0.0
    demanda_pred_media = float(np.mean(pred))
    # La incertidumbre del pronóstico se mide con la desviación de su ERROR (RMSE),
    # no con la dispersión de los valores pronosticados (que es casi plana)
    errores_pred = np.asarray(test, dtype=float) - np.asarray(pred, dtype=float)
    demanda_pred_std = float(np.sqrt(np.mean(errores_pred ** 2))) if len(errores_pred) else 0.0

    inv_real = calcular_inventario(demanda_real_media, demanda_real_std, lead_mean_days, lead_std_days, z)
    inv_pred = calcular_inventario(demanda_pred_media, demanda_pred_std, lead_mean_days, lead_std_days, z)
    
    metricas_modelo_sel = resultados["metricas"][resultados["metricas"]["Clave"] == modelo_elegido].iloc[0]

    st.markdown("---")
    st.subheader("🧮 Fórmulas de inventario")
    c1, c2, c3 = st.columns(3)
    with c1: 
        st.markdown("### Stock de Seguridad")
        st.latex(r"SS=z\sqrt{L\sigma_D^2+D^2\sigma_L^2}")
    with c2: 
        st.markdown("### Punto de Reorden")
        st.latex(r"ROP=D\cdot L+SS")
    with c3: 
        st.markdown("### Inventario Promedio")
        st.latex(r"InvProm=SS+\frac{D\cdot L}{2}")

    st.subheader("📋 Inventario requerido: real vs pronosticado")
    comparacion = pd.DataFrame({
        "Indicador": ["Demanda media semanal", "Desv. estándar (real) / Error RMSE (pronóstico)", "Lead time medio (días)", "Stock de Seguridad (SS)", "Punto de Reorden (ROP)", "Inventario Promedio Requerido"],
        "Base real": [demanda_real_media, demanda_real_std, lead_mean_days, inv_real["SS"], inv_real["ROP"], inv_real["InvProm"]],
        "Con demanda pronosticada": [demanda_pred_media, demanda_pred_std, lead_mean_days, inv_pred["SS"], inv_pred["ROP"], inv_pred["InvProm"]],
    })
    st.dataframe(comparacion.round(2), hide_index=True, use_container_width=True)

    # ---------------------------------------------------------
    # INDICADORES Y FÓRMULAS MAPE 
    # ---------------------------------------------------------
    mape_comercial = metricas_modelo_sel["MAPE %"]
    mape_operativo = (abs(inv_pred["InvProm"] - inv_real["InvProm"]) / inv_real["InvProm"] * 100) if inv_real["InvProm"] > 0 else 0.0

    st.markdown("---")
    st.subheader("🎯 Indicadores de Desviación Estratégica")
    col1, col2 = st.columns(2)
    col1.metric("MAPE Comercial (Error Demanda)", f"{mape_comercial:.2f}%")
    col2.metric("MAPE Operativo (Error Inventario)", f"{mape_operativo:.2f}%")

    with st.expander("📐 Ver fórmulas y valores de cálculo"):
        st.markdown("#### MAPE Comercial (Precisión del Pronóstico)")
        st.latex(r"MAPE_{Comercial} = \frac{1}{n} \sum_{t=1}^{n} \left| \frac{DemandaReal_t - DemandaPred_t}{DemandaReal_t} \right| \times 100")
        st.markdown(f"**Valores de cálculo ($n={len(test)}$ semanas evaluadas):**\n"
                    f"* **Demanda Real Media:** {demanda_real_media:.2f} unidades/semana.\n"
                    f"* **Demanda Pronosticada Media:** {demanda_pred_media:.2f} unidades/semana.")

        st.markdown("#### MAPE Operativo (Desviación del Inventario)")
        st.latex(r"MAPE_{Operativo} = \left| \frac{InvProm_{real} - InvProm_{pred}}{InvProm_{real}} \right| \times 100")
        st.markdown(f"**Valores de cálculo:**\n"
                    f"* **$InvProm_{{real}}$ (Basado en demanda real):** {inv_real['InvProm']:.2f} unidades.\n"
                    f"* **$InvProm_{{pred}}$ (Basado en pronóstico):** {inv_pred['InvProm']:.2f} unidades.")

    st.markdown("---")
    st.subheader("🎯 Indicadores principales")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Demanda", f"{demanda_pred_media:.1f}")
    k2.metric("Lead Time", f"{lead_mean_days:.2f} días")
    k3.metric("SS", f"{inv_pred['SS']:.1f}")
    k4.metric("ROP", f"{inv_pred['ROP']:.1f}")
    k5.metric("Inv. Prom.", f"{inv_pred['InvProm']:.1f}")

    st.markdown("---")
    st.subheader("📈 Comparación de 5 indicadores")
    indicadores = ["Demanda", "Lead Time", "SS", "ROP", "Inv. Prom."]
    real_vals = np.array([demanda_real_media, lead_mean_days, inv_real["SS"], inv_real["ROP"], inv_real["InvProm"]], dtype=float)
    pred_vals = np.array([demanda_pred_media, lead_mean_days, inv_pred["SS"], inv_pred["ROP"], inv_pred["InvProm"]], dtype=float)
    base = np.where(real_vals == 0, np.nan, real_vals)
    real_index = np.ones(len(indicadores)) * 100
    pred_index = (pred_vals / base) * 100

    fig_ind, ax_ind = plt.subplots(figsize=(11, 5))
    x_ind = np.arange(len(indicadores))
    width_ind = 0.35
    ax_ind.bar(x_ind - width_ind / 2, real_index, width_ind, label="Base real")
    ax_ind.bar(x_ind + width_ind / 2, pred_index, width_ind, label="Con demanda pronosticada")
    ax_ind.axhline(100, linewidth=1)
    ax_ind.set_xticks(x_ind)
    ax_ind.set_xticklabels(indicadores)
    ax_ind.set_ylabel("Índice (Base real = 100)")
    ax_ind.set_title(f"{categoria} · {modelo_elegido}")
    ax_ind.legend()
    ax_ind.grid(axis="y", alpha=0.25)
    st.pyplot(fig_ind)

    st.markdown("---")
    st.subheader("📐 Precisión del modelo utilizado")
    metricas_df = pd.DataFrame({
        "Métrica": ["MAD", "SSE", "MPE %", "MAPE %"],
        "Valor": [metricas_modelo_sel["MAD"], metricas_modelo_sel["SSE"], metricas_modelo_sel["MPE %"], metricas_modelo_sel["MAPE %"]]
    })
    st.dataframe(metricas_df.round(2), hide_index=True, use_container_width=True)

    # ---------------------------------------------------------
    # RIESGO DE DESABASTECIMIENTO (QUIEBRE DE STOCK)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📦 Riesgo de desabastecimiento (quiebre de stock)")
    riesgo_desab = calcular_riesgo_desabastecimiento(resultados["df_cat"], float(nivel_inv.replace("%", "")), z)
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Nivel de riesgo", riesgo_desab["nivel_riesgo"])
    d2.metric("CV de la demanda", f"{riesgo_desab['cv_demanda']*100:.1f}%")
    d3.metric("CV del lead time", f"{riesgo_desab['cv_leadtime']*100:.1f}%")
    d4.metric("Stock de seguridad (histórico)", f"{riesgo_desab['SS']:.1f} u.")
    d5.metric("Riesgo residual de quiebre", f"{riesgo_desab['riesgo_residual_pct']:.1f}%")
    st.caption(f"Clasificación según el mayor coeficiente de variación (CV = desviación / promedio) de la demanda semanal y del "
               f"lead time de toda la historia de {categoria}: bajo < 25%, medio 25–50%, alto ≥ 50%. "
               f"El riesgo residual es la probabilidad de quiebre que se acepta con un nivel de servicio de {nivel_inv}.")

    # ---------------------------------------------------------
    # ESCENARIOS DE INVENTARIO
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🧭 Escenarios de inventario")
    st.markdown(
        f"El inventario se planifica con el pronóstico de **{modelo_elegido}** y un nivel de servicio de **{nivel_inv}** "
        f"(ROP planificado = **{inv_pred['ROP']:.1f} u.**). ¿Qué pasa si el lead time real es mayor al planificado?"
    )
    meta_ns = float(stats.norm.cdf(z)) * 100

    def fila_escenario(nombre, D, sigma_D, L, sigma_L):
        req = calcular_inventario(D, sigma_D, L, sigma_L, z)
        ns = nivel_servicio_real(inv_pred["ROP"], D, sigma_D, L, sigma_L) * 100
        brecha = inv_pred["ROP"] - req["ROP"]
        if ns < meta_ns - 0.5:
            efecto = "⚠️ Riesgo de quiebre de stock"
        elif brecha > 0.5:
            efecto = "📦 Sobrestock (capital inmovilizado)"
        else:
            efecto = "✅ Inventario adecuado"
        return {"Escenario": nombre, "Demanda semanal": D, "Lead time (días)": L,
                "ROP necesario (u.)": req["ROP"], "Exceso (+) / faltante (−) vs. ROP planificado (u.)": brecha,
                "Nivel de servicio real (%)": ns, "Efecto": efecto}

    def grafico_ns(tabla, titulo):
        colores = ["#d62728" if v < meta_ns - 0.5 else "#2ca02c" for v in tabla["Nivel de servicio real (%)"]]
        fig = go.Figure(go.Bar(x=tabla["Escenario"], y=tabla["Nivel de servicio real (%)"], marker_color=colores,
                               text=tabla["Nivel de servicio real (%)"].map(lambda v: f"{v:.1f}%"), textposition="outside"))
        fig.add_hline(y=meta_ns, line_dash="dash", line_color="gray", annotation_text=f"Meta {meta_ns:.1f}%")
        fig.update_layout(title=titulo, yaxis_title="Nivel de servicio real (%)", yaxis_range=[0, 110], height=380)
        return estilo_grafico(fig)

    st.markdown("#### Escenarios logísticos: aumento del lead time")
    st.markdown(
        "Se evalúa qué pasa con el nivel de servicio si el abastecimiento se retrasa y el ROP planificado no se ajusta. "
        f"Se eligió el lead time porque es la principal fuente de incertidumbre: su coeficiente de variación es "
        f"**{riesgo_desab['cv_leadtime']*100:.1f}%**, frente a **{riesgo_desab['cv_demanda']*100:.1f}%** de la demanda."
    )
    esc_lt = pd.DataFrame([
        fila_escenario("Lead time actual", demanda_pred_media, demanda_pred_std, lead_mean_days, lead_std_days),
        fila_escenario("Lead time +1 día", demanda_pred_media, demanda_pred_std, lead_mean_days + 1, lead_std_days),
        fila_escenario("Lead time +2 días", demanda_pred_media, demanda_pred_std, lead_mean_days + 2, lead_std_days),
        fila_escenario("Variabilidad del lead time +50%", demanda_pred_media, demanda_pred_std, lead_mean_days, lead_std_days * 1.5),
    ])
    cl1, cl2 = st.columns([1.25, 1])
    with cl1:
        st.dataframe(esc_lt.round(2), hide_index=True, use_container_width=True)
    with cl2:
        st.plotly_chart(grafico_ns(esc_lt, "Nivel de servicio ante retrasos logísticos"), use_container_width=True)
    st.info(f"Un retraso de 1 día en el lead time baja el nivel de servicio a **{esc_lt.iloc[1]['Nivel de servicio real (%)']:.1f}%**; "
            f"para mantener la meta, el ROP debe subir a **{esc_lt.iloc[1]['ROP necesario (u.)']:.1f} u.** "
            f"(+{esc_lt.iloc[1]['ROP necesario (u.)'] - inv_pred['ROP']:.1f} u.).")
    st.caption("Supuesto: demanda durante el lead time con distribución normal; el ROP planificado no se actualiza en el escenario.")


# ==========================================================
# OBJETIVO 3 · ANÁLISIS DE RIESGOS
# ==========================================================
elif pagina == MENU[3]:
    st.title("⚠️ Objetivo 3 · Gestión y evaluación de riesgos")
    st.caption("Diagnóstico operativo de retrasos, severidad e inestabilidad logística.")

    df = cargar_datos(FUENTE_CSV, NOMBRE_FUENTE)
    # Se excluyen los envíos cancelados (sin resultado real de entrega)
    n_cancelados = int((df["Delivery Status"] == "Shipping canceled").sum()) if "Delivery Status" in df.columns else 0
    if n_cancelados:
        df = df[df["Delivery Status"] != "Shipping canceled"].copy()
        st.caption(f"Se excluyeron {n_cancelados:,} registros con envío cancelado, que no tienen un resultado real de entrega.")
    categorias = sorted(df["Category Name"].dropna().unique().tolist())
    regiones_disponibles = sorted(df["Order Region"].dropna().unique().tolist())
    modos_fijos = ["Todos los modos", "First Class", "Same Day", "Second Class", "Standard Class"]

    st.markdown("### 🎛️ Barra de control")
    f1, f2, f3, f4 = st.columns([1.35, 1.65, 1.25, 1.55])
    with f1: categoria = st.selectbox("Categoría de Producto", categorias, index=(categorias.index("Cleats") if "Cleats" in categorias else 0), key="riesgo_cat")
    with f2: region_sel = st.selectbox("Región de Destino", ["Todas las regiones (Consolidado Global)"] + regiones_disponibles, index=0, key="riesgo_reg")
    with f3: modo_sel = st.selectbox("Modo de Envío", modos_fijos, index=0, key="riesgo_mod")
    with f4: 
        threshold_pct = st.slider("Umbral de alerta de retraso", 50, 80, 50, 5, format="%d%%", key="riesgo_thr")
        umbral = threshold_pct / 100.0

    df_cat = df[df["Category Name"] == categoria].copy()
    if modo_sel != "Todos los modos":
        df_cat = df_cat[df_cat["Shipping Mode"] == modo_sel].copy()

    df_r_global = preparar_datos_riesgo(df_cat)
    df_r = df_r_global[df_r_global["Order Region"] == region_sel].copy() if region_sel != "Todas las regiones (Consolidado Global)" else df_r_global.copy()

    optimizar_rf = True  # Random Forest siempre optimizado con GridSearchCV
    st.caption("⏳ La primera vez que se abre cada categoría, la optimización del Random Forest (GridSearchCV) tarda alrededor de 1 minuto; luego queda guardada.")
    try:
        rf = entrenar_random_forest_riesgo(categoria, optimizar_rf)
    except Exception as e:
        st.error(f"Error entrenando Random Forest: {e}")
        st.stop()

    base_rf_global = rf["base"].copy()
    base_rf_global["Prediccion_Riesgo_Retraso"] = (pd.to_numeric(base_rf_global["Probabilidad_Retraso"], errors="coerce") >= umbral).astype(int)

    if modo_sel != "Todos los modos":
        base_rf_global = base_rf_global[base_rf_global["Shipping Mode"] == modo_sel].copy()

    base_rf = base_rf_global[base_rf_global["Order Region"] == region_sel].copy() if region_sel != "Todas las regiones (Consolidado Global)" else base_rf_global.copy()

    st.markdown("---")
    st.subheader("🚨 Riesgo general de la cadena de suministro")
    tasa_riesgo = float(base_rf["Prediccion_Riesgo_Retraso"].mean()) if len(base_rf) else 0.0
    prob_media = float(base_rf["Probabilidad_Retraso"].mean()) if len(base_rf) else 0.0
    severidad_media = float(base_rf["Retraso_Dias"].mean()) if len(base_rf) else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pedidos en riesgo", f"{tasa_riesgo*100:.1f}%")
    k2.metric("Probabilidad media", f"{prob_media*100:.1f}%")
    k3.metric("Severidad promedio", f"{severidad_media:.2f} días")
    k4.metric("Exactitud Modelo (Accuracy)", f"{rf['metricas']['Accuracy']*100:.1f}%")

    # ---------------------------------------------------------
    # DESEMPEÑO DEL MODELO RANDOM FOREST
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📐 Desempeño del modelo Random Forest")
    st.caption(f"Entrenamiento: {rf['n_train']:,} registros · Prueba: {rf['n_test']:,} registros · "
               "División por pedido (Order Id) para que los ítems de un mismo pedido no aparezcan en ambos grupos.")
    p_rf = rf["modelo"].named_steps["rf"].get_params()
    st.markdown(
        f"**Modelo en uso ({'optimizado con GridSearchCV' if optimizar_rf else 'configuración original'}):** "
        f"🌳 **{p_rf['n_estimators']} árboles** · profundidad máxima: **{p_rf['max_depth'] if p_rf['max_depth'] is not None else 'sin límite'}** · "
        f"mínimo de registros por hoja: **{p_rf['min_samples_leaf']}** · criterio: **{p_rf['criterion']}**"
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{rf['metricas']['Accuracy']*100:.1f}%")
    m2.metric("Precision", f"{rf['metricas']['Precision']*100:.1f}%")
    m3.metric("Recall", f"{rf['metricas']['Recall']*100:.1f}%")
    m4.metric("F1-score", f"{rf['metricas']['F1']*100:.1f}%")
    m5.metric("ROC AUC", f"{rf['metricas']['ROC_AUC']:.3f}")

    col_cm, col_roc = st.columns(2)
    with col_cm:
        cm = rf["cm"]
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=["Predicho: A tiempo", "Predicho: Tardío"], y=["Real: A tiempo", "Real: Tardío"],
            title="Matriz de confusión (datos de prueba)",
        )
        fig_cm.update_layout(coloraxis_showscale=False, height=420)
        st.plotly_chart(estilo_grafico(fig_cm), use_container_width=True)
        st.caption(f"Aciertos: {cm[0, 0]:,} a tiempo y {cm[1, 1]:,} tardíos · "
                   f"Detecta el {rf['metricas']['Recall']*100:.1f}% de los retrasos reales.")
    with col_roc:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=rf["fpr"], y=rf["tpr"], mode="lines", name=f"Random Forest (AUC = {rf['metricas']['ROC_AUC']:.3f})", line=dict(color="#6A3D9A", width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Referencia aleatoria (AUC = 0.5)", line=dict(color="gray", dash="dash")))
        fig_roc.update_layout(title="Curva ROC", xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos (Recall)", height=420, legend=dict(x=0.35, y=0.08))
        st.plotly_chart(estilo_grafico(fig_roc), use_container_width=True)

    st.markdown("**Comparación con una regla simple basada solo en el modo de envío**")
    comp_regla = pd.DataFrame({
        "Métrica": ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"],
        "Random Forest": [rf["metricas"][k] for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]],
        "Regla por modo de envío": [rf["metricas_regla"][k] for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]],
    })
    comp_regla["Diferencia"] = comp_regla["Random Forest"] - comp_regla["Regla por modo de envío"]
    st.dataframe(comp_regla.round(4), hide_index=True, use_container_width=True)
    st.info("Si ambos resultados son similares, el riesgo de retraso es principalmente estructural: depende del modo de "
            "envío y de su plazo prometido, más que de otros factores del pedido (en línea con Saini et al., 2026).")

    # ---------------------------------------------------------
    # OPTIMIZACIÓN DE HIPERPARÁMETROS
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🔧 Optimización de hiperparámetros (GridSearchCV)")
    if optimizar_rf and rf["mejores_params"] is not None:
        rf_orig = entrenar_random_forest_riesgo(categoria, False)
        st.markdown(
            "Se probaron **18 combinaciones** de `n_estimators` (100, 200), `max_depth` (sin límite, 8, 16) y "
            "`min_samples_leaf` (1, 20, 50), con validación cruzada de **3 particiones agrupadas por pedido**, eligiendo la de "
            "mayor ROC AUC promedio."
        )
        p_orig = rf_orig["modelo"].named_steps["rf"].get_params()
        p_opt = rf["modelo"].named_steps["rf"].get_params()
        fmt_prof = lambda v: "Sin límite" if v is None else str(v)
        h1, h2, h3 = st.columns(3)
        h1.metric("🌳 Número de árboles (n_estimators)", f"{p_opt['n_estimators']}",
                  f"{p_opt['n_estimators'] - p_orig['n_estimators']:+d} vs. original ({p_orig['n_estimators']})", delta_color="off")
        h2.metric("Profundidad máxima (max_depth)", fmt_prof(p_opt["max_depth"]),
                  f"Original: {fmt_prof(p_orig['max_depth'])}", delta_color="off")
        h3.metric("Mínimo por hoja (min_samples_leaf)", f"{p_opt['min_samples_leaf']}",
                  f"Original: {p_orig['min_samples_leaf']}", delta_color="off")
        st.success(f"Con la optimización, el modelo usa **{p_opt['n_estimators']} árboles** (antes {p_orig['n_estimators']}), "
                   f"con profundidad máxima de **{fmt_prof(p_opt['max_depth'])}** y al menos **{p_opt['min_samples_leaf']}** registros por hoja. "
                   "Árboles más simples y menos numerosos reducen el sobreajuste y el tiempo de cálculo.")
        comp_rf = pd.DataFrame({
            "Métrica": ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"],
            "Original (200 árboles, sin límite)": [rf_orig["metricas"][k] for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]],
            "Optimizado (GridSearchCV)": [rf["metricas"][k] for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]],
        })
        comp_rf["Diferencia"] = comp_rf["Optimizado (GridSearchCV)"] - comp_rf["Original (200 árboles, sin límite)"]
        cg1, cg2 = st.columns(2)
        with cg1:
            st.markdown("**Original vs. optimizado (datos de prueba)**")
            st.dataframe(comp_rf.round(4), hide_index=True, use_container_width=True)
        with cg2:
            st.markdown("**Mejores 5 combinaciones (validación cruzada)**")
            st.dataframe(rf["tabla_grid"].head(5).round(4), hide_index=True, use_container_width=True)

    # ---------------------------------------------------------
    # VARIABLE MÁS IMPORTANTE
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🔑 Variable más importante para predecir el retraso")
    imp = rf["importancia"]
    ci1, ci2 = st.columns([1.2, 1])
    with ci1:
        fig_imp = px.bar(imp.sort_values("Importancia Gini (%)"), x="Importancia Gini (%)", y="Variable", orientation="h",
                         text="Importancia Gini (%)", color="Importancia Gini (%)", color_continuous_scale="Purples",
                         title="Importancia de variables (Random Forest)")
        fig_imp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_imp.update_layout(coloraxis_showscale=False, yaxis_title="", height=380)
        st.plotly_chart(estilo_grafico(fig_imp), use_container_width=True)
    with ci2:
        st.dataframe(imp.round(4), hide_index=True, use_container_width=True)
        top = imp.iloc[0]
        st.success(f"La variable más importante es **{top['Variable']}** ({top['Importancia Gini (%)']:.1f}% de la importancia). "
                   f"Si se desordena al azar, el ROC AUC cae {top['Caída de ROC AUC al permutar']:.3f}.")
    st.caption("Importancia Gini: cuánto ayuda cada variable a separar pedidos tardíos y puntuales dentro de los árboles. "
               "Permutación: cuánto empeora el modelo en los datos de prueba si esa variable se mezcla al azar. "
               "El PCA no sirve para esto, porque no usa la variable de retraso y no incluye variables categóricas como el modo de envío.")

    st.markdown("---")
    st.subheader("🗺️ Riesgo de retraso por región")
    region_risk = base_rf_global.groupby("Order Region").agg(Pedidos_Totales=("Order Region", "size"), Pedidos_Riesgo=("Prediccion_Riesgo_Retraso", "sum"), Probabilidad=("Probabilidad_Retraso", "mean"), Severidad=("Retraso_Dias", "mean")).reset_index()
    region_risk["Tasa de Riesgo (%)"] = (region_risk["Pedidos_Riesgo"] / region_risk["Pedidos_Totales"]) * 100

    if region_sel == "Todas las regiones (Consolidado Global)":
        fig_region = px.bar(region_risk.sort_values("Pedidos_Riesgo", ascending=True), x="Pedidos_Riesgo", y="Order Region", orientation="h", color="Tasa de Riesgo (%)", text="Pedidos_Riesgo", color_continuous_scale=[(0.0, "#2ca02c"), (0.40, "#2ca02c"), (0.70, "#f1c40f"), (1.0, "#d62728")])
    else:
        seleccionado = region_risk[region_risk["Order Region"] == region_sel].copy()
        promedio = pd.DataFrame([{"Order Region": "Promedio global", "Pedidos_Riesgo": region_risk["Pedidos_Riesgo"].mean(), "Tasa de Riesgo (%)": region_risk["Tasa de Riesgo (%)"].mean()}])
        fig_region = px.bar(pd.concat([seleccionado, promedio], ignore_index=True), x="Order Region", y="Pedidos_Riesgo", color="Tasa de Riesgo (%)", text="Pedidos_Riesgo", color_continuous_scale=[(0.0, "#2ca02c"), (0.40, "#2ca02c"), (0.70, "#f1c40f"), (1.0, "#d62728")])

    fig_region.update_traces(textposition="outside")
    fig_region.update_layout(height=560, coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(estilo_grafico(fig_region), use_container_width=True)

    st.markdown("---")
    st.subheader("🧩 Riesgo por zona · Perfiles (K-Means)")
    # Se usan todos los modos de envío de la categoría: con un solo modo (p. ej. First Class) todas las regiones
    # tienen el mismo comportamiento y no hay grupos que formar
    df_r_km = preparar_datos_riesgo(df[df["Category Name"] == categoria])
    st.caption(f"Perfil de las regiones con todos los modos de envío de {categoria} (no depende del filtro de modo de envío).")
    try:
        km_resumen = ejecutar_kmeans_riesgo(df_r_km, n_clusters=3)
        fig_km = px.scatter(km_resumen, x="Probabilidad_Retraso", y="Severidad_Promedio", size="Pedidos_Riesgo", color="Diagnostico", text="Order Region", hover_name="Order Region", size_max=50, color_discrete_map={"🟢 Clúster 0 · Flujo Estable": "#2ca02c", "🟡 Clúster 1 · Retraso Frecuente pero Leve": "#f1c40f", "🔴 Clúster 2 · Zona Crítica": "#d62728"})
        fig_km.update_traces(textposition="top center", marker={"line": {"width": 1, "color": "white"}})
        fig_km.update_layout(height=540, xaxis_tickformat=".0%")
        st.plotly_chart(estilo_grafico(fig_km), use_container_width=True)

        # Optimización del número de clústeres: codo y silueta
        eval_k = evaluar_k_regiones(df_r_km)
        if len(eval_k):
            ck1, ck2 = st.columns(2)
            with ck1:
                fig_codo = px.line(eval_k, x="k", y="Inercia", markers=True, title="Método del codo")
                fig_codo.add_vline(x=3, line_dash="dash", line_color="gray")
                st.plotly_chart(estilo_grafico(fig_codo), use_container_width=True)
            with ck2:
                fig_sil = px.line(eval_k, x="k", y="Silueta", markers=True, title="Coeficiente de silueta por k")
                fig_sil.add_vline(x=3, line_dash="dash", line_color="gray")
                st.plotly_chart(estilo_grafico(fig_sil), use_container_width=True)
            k_best = int(eval_k.loc[eval_k["Silueta"].idxmax(), "k"])
            sil3 = eval_k.loc[eval_k["k"] == 3, "Silueta"]
            st.caption(f"La silueta más alta se obtiene con k = {k_best} ({eval_k['Silueta'].max():.3f}); con k = 3 es "
                       f"{(sil3.iloc[0] if len(sil3) else float('nan')):.3f}. Se mantiene k = 3 por criterio de negocio: "
                       "tres niveles de riesgo (estable, leve y crítico) fáciles de convertir en acciones.")
    except Exception as e:
        st.warning(f"No fue posible ejecutar K-Means: {e}")

    st.markdown("---")
    st.subheader("🎯 Riesgo combinado: probabilidad vs. severidad")
    matriz_base = calcular_matriz_operativa(base_rf)
    if len(matriz_base):
        matriz_base["Zona"] = [("🔴 Cuadrante Crítico" if (p >= umbral and s > 1.0) else "🟡 Cuadrante Moderado" if (p >= umbral and s <= 1.0) else "🟨 Cuadrante Ocasional" if (p < umbral and s > 1.0) else "🟢 Cuadrante Controlado") for p, s in zip(matriz_base["Probabilidad_Retraso"], matriz_base["Severidad_Dias"])]
        fig_matrix = px.scatter(matriz_base, x="Probabilidad_Retraso", y="Severidad_Dias", size="Pedidos", color="Zona", text="Order Region", size_max=48, color_discrete_map={"🔴 Cuadrante Crítico": "#d62728", "🟡 Cuadrante Moderado": "#f1c40f", "🟨 Cuadrante Ocasional": "#e67e22", "🟢 Cuadrante Controlado": "#2ca02c"})
        fig_matrix.add_vline(x=umbral, line_dash="dash", line_width=2, annotation_text=f"Umbral {umbral:.0%}")
        fig_matrix.add_hline(y=1.0, line_dash="dash", line_width=2, annotation_text="Severidad > 1 día")
        fig_matrix.update_layout(height=520, xaxis_tickformat=".0%")
        st.plotly_chart(estilo_grafico(fig_matrix), use_container_width=True)

    st.markdown("---")
    st.subheader("🚚 Riesgo de cumplimiento de llegada (On-Time)")
    lt_base = df_r.dropna(subset=["Days for shipping (real)", "Days for shipment (scheduled)"]).copy()
    confiabilidad = float(lt_base["Cumple_LeadTime"].mean())
    c1, c2, c3 = st.columns(3)
    c1.metric("Cumplimiento del plazo", f"{confiabilidad*100:.1f}%")
    c2.metric("Exposición por incumplimiento", f"{(1-confiabilidad)*100:.1f}%")
    c3.metric("Exceso P95", f"{float(lt_base['Retraso_Dias'].quantile(0.95) if len(lt_base) else 0):.2f} días")
    
    fig_lt = go.Figure()
    fig_lt.add_trace(go.Box(y=lt_base["Days for shipping (real)"], name="Días reales", boxpoints="outliers"))
    fig_lt.add_trace(go.Box(y=lt_base["Days for shipment (scheduled)"], name="Días programados", boxpoints=False))
    st.plotly_chart(estilo_grafico(fig_lt), use_container_width=True)


    # ---------------------------------------------------------
    # ANÁLISIS PRESCRIPTIVO: RECALIBRACIÓN DE PROMESAS DE ENTREGA
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🧭 Análisis prescriptivo · Recalibración del plazo prometido por modo de envío")

    alcance = st.radio("Alcance del análisis", ["Toda la empresa", f"Solo la categoría {categoria}"], horizontal=True, key="riesgo_alcance")
    base_esc = df if alcance == "Toda la empresa" else df[df["Category Name"] == categoria]

    por_modo = base_esc.dropna(subset=["Days for shipping (real)", "Days for shipment (scheduled)"]).groupby("Shipping Mode").agg(
        Pedidos=("Shipping Mode", "size"),
        Plazo_prometido=("Days for shipment (scheduled)", "mean"),
        Plazo_real_promedio=("Days for shipping (real)", "mean"),
        Pct_tarde=("Late_delivery_risk", "mean"),
    ).reset_index()
    por_modo["Pct_tarde"] = por_modo["Pct_tarde"] * 100
    por_modo.columns = ["Modo de envío", "Pedidos", "Plazo prometido (días)", "Plazo real promedio (días)", "% pedidos tarde"]
    st.markdown("**Diagnóstico actual por modo de envío**")
    st.dataframe(por_modo.round(2), hide_index=True, use_container_width=True)

    st.markdown("""
**¿Por qué estos escenarios?** El diagnóstico muestra que el retraso no es aleatorio: depende del modo de envío.
**First Class** promete 1 día pero siempre demora 2 (100% tarde), **Second Class** promete 2 y demora casi 4 (≈80% tarde)
y **Same Day** promete 0 y a veces demora 1 día. **Standard Class** promete 4 y cumple en promedio, por eso no se modifica.
Cada escenario ajusta la promesa de un modo más, empezando por el que falla siempre, para igualarla con lo que
la operación realmente logra.
""")
    tabla_esc = escenarios_recalibracion(base_esc)

    # % de pedidos tarde por modo de envío en cada escenario
    base_modo = base_esc.dropna(subset=["Days for shipping (real)", "Days for shipment (scheduled)", "Shipping Mode"])
    cambios_esc = {"Actual": {}, "E1": {"First Class": 2}, "E2": {"First Class": 2, "Second Class": 4},
                   "E3": {"First Class": 2, "Second Class": 4, "Same Day": 1}}
    filas_modo = []
    for modo_env, grupo in base_modo.groupby("Shipping Mode"):
        fila = {"Modo de envío": modo_env, "% de los pedidos": len(grupo) / len(base_modo) * 100}
        for esc, cambios in cambios_esc.items():
            prometido = cambios.get(modo_env, None)
            p = grupo["Days for shipment (scheduled)"] if prometido is None else prometido
            fila[f"% tarde · {esc}"] = (grupo["Days for shipping (real)"] > p).mean() * 100
        filas_modo.append(fila)
    tabla_modo = pd.DataFrame(filas_modo)
    st.markdown("**% de pedidos tarde por modo de envío en cada escenario**")
    st.dataframe(tabla_modo.round(1), hide_index=True, use_container_width=True)

    peso_fc = tabla_modo.loc[tabla_modo["Modo de envío"] == "First Class", "% de los pedidos"]
    peso_fc = float(peso_fc.iloc[0]) if len(peso_fc) else 0.0
    st.info(f"**¿Por qué First Class pasa de 100% a 0% tarde, pero el total solo baja {tabla_esc['Reducción vs. actual (p.p.)'].iloc[1]:.1f} puntos?** "
            f"Porque el % de pedidos tarde se calcula sobre **todos** los pedidos, y First Class representa solo el "
            f"**{peso_fc:.1f}%** de ellos. Al corregir su promesa, ese {peso_fc:.1f}% deja de llegar tarde: "
            f"el total baja aproximadamente {peso_fc:.1f} puntos, de {tabla_esc['% pedidos tarde'].iloc[0]:.1f}% a "
            f"{tabla_esc['% pedidos tarde'].iloc[1]:.1f}%. Los demás modos siguen igual en ese escenario.")

    col_e1, col_e2 = st.columns([1.1, 1])
    with col_e1:
        st.markdown("**Escenarios evaluados**")
        st.dataframe(tabla_esc.round(2), hide_index=True, use_container_width=True)
    with col_e2:
        fig_esc = px.bar(tabla_esc, x="Escenario", y="% pedidos tarde", text="% pedidos tarde",
                         color="% pedidos tarde", color_continuous_scale=[(0, "#2ca02c"), (0.5, "#f1c40f"), (1, "#d62728")],
                         title="% de pedidos tardíos por escenario")
        fig_esc.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_esc.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="% pedidos tarde", height=420)
        st.plotly_chart(estilo_grafico(fig_esc), use_container_width=True)

    mejor = tabla_esc.iloc[-1]
    st.success(f"Ajustar los plazos prometidos reduciría los pedidos tardíos de {tabla_esc['% pedidos tarde'].iloc[0]:.1f}% "
               f"a {mejor['% pedidos tarde']:.1f}% ({mejor['Reducción vs. actual (p.p.)']:.1f} puntos porcentuales), "
               "sin cambiar la operación de despacho: el retraso proviene principalmente de promesas que hoy no se pueden cumplir.")
    st.caption("Supuesto: los días reales de envío se mantienen; solo cambia el plazo comunicado al cliente para cada modo.")


# ==========================================================
# OBJETIVO 4 · PRICING
# ==========================================================
elif pagina == MENU[4]:
    st.title("💰 Objetivo 4 · Pricing y Elasticidad")
    st.markdown("**Analizar la respuesta de la demanda al precio y a los descuentos (Log-Log), y evaluar escenarios de política de descuentos.**")

    df = cargar_datos(FUENTE_CSV, NOMBRE_FUENTE)

    # 1. Preparación de datos operativos
    df_trans = df.copy()
    df_trans["Producto_Match"] = df_trans["Product Name"].astype(str).str.replace("’", "'").str.strip()
    df_trans = df_trans[df_trans["Order Item Quantity"] > 0]
    df_trans = df_trans[df_trans["Sales"] > 0]
    df_trans["Discount"] = df_trans["Order Item Discount"].fillna(0)
    df_trans["Precio_Neto"] = round((df_trans["Sales"] - df_trans["Discount"]) / df_trans["Order Item Quantity"], 2)
    df_trans = df_trans[df_trans["Precio_Neto"] > 0]

    # ---------------------------------------------------------
    # PARETO (TOP 10 GLOBALES POR VENTAS)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🎯 Análisis de Pareto: Top Productos Clave")
    ventas_totales = df_trans["Sales"].sum()
    datos_pareto = df_trans.groupby("Producto_Match")["Sales"].sum().reset_index()
    datos_pareto = datos_pareto.sort_values("Sales", ascending=False).head(10).reset_index(drop=True)
    # Porcentaje sobre las ventas TOTALES de la empresa
    datos_pareto["Porcentaje"] = (datos_pareto["Sales"] / ventas_totales) * 100
    datos_pareto["Acumulado"] = datos_pareto["Porcentaje"].cumsum()
    datos_pareto["Nombre_Limpio"] = datos_pareto["Producto_Match"].str[:22]

    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pareto.add_trace(
        go.Bar(x=datos_pareto["Nombre_Limpio"], y=datos_pareto["Sales"], name="Ventas Totales ($)", marker_color="#1F77B4"),
        secondary_y=False
    )
    fig_pareto.add_trace(
        go.Scatter(x=datos_pareto["Nombre_Limpio"], y=datos_pareto["Acumulado"], name="% Acumulado", mode="lines+markers", line=dict(color="#D62728", width=3), marker=dict(size=8)),
        secondary_y=True
    )
    fig_pareto.add_hline(y=80, line_dash="dash", line_color="gray", secondary_y=True)
    fig_pareto.update_layout(title_text="Concentración de Ventas (Top 10) sobre el total de la empresa", xaxis_tickangle=-45, legend=dict(x=0.01, y=0.99))
    fig_pareto.update_yaxes(title_text="Ventas Totales ($)", secondary_y=False)
    fig_pareto.update_yaxes(title_text="% Acumulado del total de ventas", range=[0, 105], secondary_y=True)
    st.plotly_chart(estilo_grafico(fig_pareto), use_container_width=True)

    # Los 8 productos con mayores ventas (se toman de los datos, con su nombre exacto)
    top_8_pareto = datos_pareto["Producto_Match"].head(8).tolist()
    st.caption(f"Los 8 productos principales concentran el **{datos_pareto['Acumulado'].iloc[7]:.1f}%** de las ventas totales.")

    # ---------------------------------------------------------
    # CÁLCULO DE ELASTICIDAD ECONOMÉTRICA (TOP 8)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Elasticidad Precio de la Demanda (Log-Log)")
    st.latex(r"\ln(Q) = \beta_0 + \beta_1 \ln(P) \qquad E_p = \beta_1")

    df_top8 = df_trans[df_trans["Producto_Match"].isin(top_8_pareto)]
    curvas = df_top8.groupby(["Producto_Match", "Precio_Neto"])["Order Item Quantity"].sum().reset_index()

    resultados_elasticidad = []
    for prod in top_8_pareto:
        curva_prod = curvas[curvas["Producto_Match"] == prod]
        est = estimar_elasticidad(curva_prod["Precio_Neto"], curva_prod["Order Item Quantity"])
        resultados_elasticidad.append({
            "Producto_Match": prod,
            "Elasticidad": est["b1"],
            "Intercepto": est["b0"],
            "R2": est["R2"],
            "p_valor": est["p_valor"],
            "Niveles de precio": est["n"],
            "Tipo": clasificar_elasticidad(est["b1"], est["p_valor"]),
            "Nombre_Limpio": prod[:26] + ("..." if len(prod) > 26 else ""),
        })

    df_resultados = pd.DataFrame(resultados_elasticidad).sort_values("Elasticidad").reset_index(drop=True)

    colores_tipo = {
        "Elástica (Sensible)": "#2CA02C",
        "Inelástica (Poco sensible)": "#1F77B4",
        "No significativa (p ≥ 0.05)": "#7F7F7F",
        "Sin variación de precio": "#BBBBBB",
        "Atípica (pendiente positiva)": "#D62728",
    }

    fig_elast = px.bar(
        df_resultados,
        y="Nombre_Limpio",
        x="Elasticidad",
        color="Tipo",
        color_discrete_map=colores_tipo,
        orientation="h",
        title="Coeficiente de Elasticidad Precio (Ep) sobre Transacciones Reales"
    )
    fig_elast.add_vline(x=-1, line_dash="dash", line_color="black")
    fig_elast.update_traces(texttemplate='%{x:.3f}', textposition='outside')
    fig_elast.update_layout(xaxis_title="Coeficiente de Elasticidad (Ep)", yaxis_title="Producto")
    st.plotly_chart(estilo_grafico(fig_elast), use_container_width=True)

    tabla_elast = df_resultados[["Producto_Match", "Elasticidad", "R2", "p_valor", "Niveles de precio", "Tipo"]].copy()
    tabla_elast.columns = ["Producto", "Elasticidad (β1)", "R²", "p-valor", "Niveles de precio", "Clasificación"]
    st.dataframe(tabla_elast.round(4), hide_index=True, use_container_width=True)

    n_signif = int((df_resultados["p_valor"] < 0.05).sum())
    if n_signif == 0:
        st.warning(
            "Ninguna elasticidad es estadísticamente significativa (p ≥ 0.05): la cantidad vendida no responde a las "
            "variaciones del precio neto. Cada producto tiene un único precio de lista y el precio neto solo varía por "
            "los descuentos, por lo que a continuación se evalúa directamente la efectividad de los descuentos."
        )
    else:
        st.info(f"{n_signif} de {len(df_resultados)} productos presentan una elasticidad significativa al 5%.")

    # ---------------------------------------------------------
    # CURVAS DE DEMANDA (TOP 3 CON ELASTICIDAD MÁS NEGATIVA)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📉 Curvas de Demanda: Top 3 Productos con elasticidad más negativa")

    top_3_elasticos = df_resultados.dropna(subset=["Elasticidad"]).head(3)["Producto_Match"].tolist()

    fig_curvas = make_subplots(rows=1, cols=max(len(top_3_elasticos), 1), subplot_titles=[p[:24] + "..." for p in top_3_elasticos])

    for i, prod in enumerate(top_3_elasticos):
        df_prod = curvas[curvas["Producto_Match"] == prod]
        fila = df_resultados[df_resultados["Producto_Match"] == prod].iloc[0]
        p = df_prod["Precio_Neto"].values
        q = df_prod["Order Item Quantity"].values

        p_range = np.linspace(p.min() * 0.85, p.max() * 1.15, 60)
        q_est = np.exp(fila["Intercepto"] + fila["Elasticidad"] * np.log(p_range))

        fig_curvas.add_trace(
            go.Scatter(x=p, y=q, mode="markers", marker=dict(color="#1F77B4", size=8, opacity=0.6), showlegend=False),
            row=1, col=i + 1
        )
        fig_curvas.add_trace(
            go.Scatter(x=p_range, y=q_est, mode="lines", line=dict(color="#D62728", width=3), showlegend=False),
            row=1, col=i + 1
        )
        fig_curvas.update_xaxes(title_text="Precio Neto ($)", row=1, col=i + 1)
        if i == 0:
            fig_curvas.update_yaxes(title_text="Cantidad Demandada", row=1, col=i + 1)

    fig_curvas.update_layout(title_text="Puntos de venta reales (Azul) vs. Curva econométrica ajustada (Rojo)", height=400)
    st.plotly_chart(estilo_grafico(fig_curvas), use_container_width=True)

    # ---------------------------------------------------------
    # EFECTIVIDAD DE LOS DESCUENTOS
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🏷️ Efectividad de los descuentos en los productos principales")

    df_desc = df_top8.copy()
    df_desc["Banda de descuento"] = pd.cut(
        df_desc["Order Item Discount Rate"], [-0.001, 0.05, 0.10, 0.15, 0.25],
        labels=["0–5%", "6–10%", "11–15%", "16–25%"]
    )
    bandas = df_desc.groupby("Banda de descuento", observed=True).agg(
        Lineas=("Sales", "size"),
        Cantidad_promedio=("Order Item Quantity", "mean"),
        Descuento_promedio=("Discount", "mean"),
        Beneficio_promedio=("Benefit per order", "mean"),
        Beneficio_total=("Benefit per order", "sum"),
        Ventas_total=("Sales", "sum"),
    ).reset_index()
    bandas["Margen (%)"] = bandas["Beneficio_total"] / bandas["Ventas_total"] * 100
    bandas = bandas.drop(columns=["Beneficio_total", "Ventas_total"])
    bandas.columns = ["Banda de descuento", "Líneas de pedido", "Cantidad promedio (u.)", "Descuento promedio ($)", "Beneficio promedio ($)", "Margen (%)"]

    rho_q, p_q = stats.spearmanr(df_desc["Order Item Discount Rate"], df_desc["Order Item Quantity"])
    rho_b, p_b = stats.spearmanr(df_desc["Order Item Discount Rate"], df_desc["Benefit per order"])
    grupos = [g["Order Item Quantity"].values for _, g in df_desc.groupby("Banda de descuento", observed=True)]
    p_kw = stats.kruskal(*grupos).pvalue if len(grupos) > 1 else np.nan

    col_b1, col_b2 = st.columns([1.15, 1])
    with col_b1:
        st.dataframe(bandas.round(2), hide_index=True, use_container_width=True)
        pruebas = pd.DataFrame({
            "Prueba": ["Spearman: descuento vs. cantidad", "Spearman: descuento vs. beneficio", "Kruskal-Wallis: cantidad entre bandas"],
            "Estadístico": [rho_q, rho_b, np.nan],
            "p-valor": [p_q, p_b, p_kw],
        })
        pruebas["Conclusión"] = np.where(pruebas["p-valor"] < 0.05, "Relación / diferencia significativa", "Sin relación / sin diferencia")
        st.dataframe(pruebas.round(4), hide_index=True, use_container_width=True)
    with col_b2:
        fig_b = make_subplots(specs=[[{"secondary_y": True}]])
        fig_b.add_trace(go.Bar(x=bandas["Banda de descuento"], y=bandas["Beneficio promedio ($)"], name="Beneficio promedio ($)", marker_color="#2CA02C"), secondary_y=False)
        fig_b.add_trace(go.Scatter(x=bandas["Banda de descuento"], y=bandas["Cantidad promedio (u.)"], name="Cantidad promedio (u.)", mode="lines+markers", line=dict(color="#D62728", width=3)), secondary_y=True)
        fig_b.update_layout(title_text="Más descuento: menos beneficio, misma cantidad", legend=dict(orientation="h", y=-0.2))
        fig_b.update_yaxes(title_text="Beneficio promedio ($)", secondary_y=False)
        fig_b.update_yaxes(title_text="Cantidad promedio (u.)", range=[0, max(3, bandas["Cantidad promedio (u.)"].max() * 1.3)], secondary_y=True)
        st.plotly_chart(estilo_grafico(fig_b), use_container_width=True)

    descuento_total = df_desc["Discount"].sum()
    st.info(f"En los 8 productos principales se otorgaron **${descuento_total:,.0f}** en descuentos. "
            f"La cantidad por línea de pedido no cambia entre bandas (p = {p_kw:.3f}), mientras que el beneficio disminuye "
            f"a mayor descuento (p = {p_b:.2g}).")

    # ---------------------------------------------------------
    # ANÁLISIS PRESCRIPTIVO: TOPE DE DESCUENTO
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🧭 Análisis prescriptivo · Escenarios de tope de descuento")
    st.markdown(
        "Un tope de descuento **sube el precio neto** que paga el cliente. Cuánto cambia la cantidad vendida lo dice la "
        "**elasticidad precio** estimada arriba: cantidad nueva = cantidad actual × (precio nuevo / precio actual)^Ep. "
        "Con elasticidades cercanas a 0, la cantidad prácticamente no cambia y el descuento eliminado se convierte en beneficio."
    )

    e_por_producto = df_resultados.set_index("Producto_Match")["Elasticidad"].fillna(0.0)

    def beneficio_con_tope(tope, elasticidad=None):
        """Recalcula descuento, cantidad y beneficio de cada línea de pedido con un tope de descuento."""
        tasa_0 = df_desc["Order Item Discount Rate"].clip(0, 0.99)
        tasa_1 = np.minimum(tasa_0, tope)
        costo = df_desc["Sales"] - df_desc["Discount"] - df_desc["Benefit per order"]
        e = df_desc["Producto_Match"].map(e_por_producto).fillna(0.0) if elasticidad is None else elasticidad
        factor_q = ((1 - tasa_1) / (1 - tasa_0)) ** e          # respuesta de la cantidad al nuevo precio neto
        ingreso_neto = df_desc["Sales"] * (1 - tasa_1) * factor_q
        beneficio = ingreso_neto - costo * factor_q
        return {
            "descuento": float((df_desc["Sales"] * tasa_1 * factor_q).sum()),
            "cantidad": float((df_desc["Order Item Quantity"] * factor_q).sum()),
            "beneficio": float(beneficio.sum()),
        }

    beneficio_actual = df_desc["Benefit per order"].sum()
    cantidad_actual = df_desc["Order Item Quantity"].sum()
    TOPE_IDEAL = 0.10
    escenarios_tope = [
        ("Tope 20%", 0.20, "Menos restrictivo: gana menos"),
        ("Tope 15%", 0.15, "Menos restrictivo: gana menos"),
        ("Tope 10% ⭐ IDEAL", 0.10, "Equilibrio: mayor beneficio afectando a menos de la mitad de los pedidos"),
        ("Tope 5%", 0.05, "Más restrictivo: gana más, pero afecta a la mayoría de clientes"),
        ("Sin descuentos", 0.0, "Máximo teórico: casi todos los clientes pierden su descuento"),
    ]
    filas_esc = [{"Escenario": "Actual (hoy)", "Tope de descuento": "Sin tope", "Descuento otorgado ($)": descuento_total,
                  "Pedidos afectados (%)": 0.0, "Descuento que pierde cada cliente afectado ($)": 0.0,
                  "Cambio en cantidad (%)": 0.0, "Beneficio ($)": beneficio_actual, "Variación del beneficio (%)": 0.0,
                  "Lectura": "Punto de partida"}]
    for nombre, tope, lectura in escenarios_tope:
        r = beneficio_con_tope(tope)
        afectados = df_desc["Order Item Discount Rate"] > tope + 1e-9
        perdida = (df_desc["Discount"] - np.minimum(df_desc["Order Item Discount Rate"], tope) * df_desc["Sales"])[afectados]
        filas_esc.append({
            "Escenario": nombre, "Tope de descuento": f"{tope:.0%}", "Descuento otorgado ($)": r["descuento"],
            "Pedidos afectados (%)": afectados.mean() * 100,
            "Descuento que pierde cada cliente afectado ($)": float(perdida.mean()) if afectados.any() else 0.0,
            "Cambio en cantidad (%)": (r["cantidad"] / cantidad_actual - 1) * 100,
            "Beneficio ($)": r["beneficio"],
            "Variación del beneficio (%)": (r["beneficio"] / beneficio_actual - 1) * 100 if beneficio_actual else np.nan,
            "Lectura": lectura,
        })
    tabla_desc = pd.DataFrame(filas_esc)
    fila_ideal = tabla_desc[tabla_desc["Escenario"].str.contains("IDEAL")].iloc[0]

    st.dataframe(tabla_desc.round(2), hide_index=True, use_container_width=True)
    st.caption("**Actual (hoy)** es el punto de partida, por eso su variación es 0%. **Pedidos afectados**: líneas de pedido "
               "cuyo descuento actual supera el tope y que, por lo tanto, recibirían un descuento menor.")

    colores_esc = ["#7F7F7F", "#AEC7E8", "#6BAED6", "#2CA02C", "#FF7F0E", "#D62728"]
    fig_d = make_subplots(specs=[[{"secondary_y": True}]])
    fig_d.add_trace(go.Bar(x=tabla_desc["Escenario"], y=tabla_desc["Beneficio ($)"], marker_color=colores_esc,
                           text=tabla_desc["Variación del beneficio (%)"].map(lambda v: f"{v:+.1f}%"), textposition="outside",
                           name="Beneficio ($)"), secondary_y=False)
    fig_d.add_trace(go.Scatter(x=tabla_desc["Escenario"], y=tabla_desc["Pedidos afectados (%)"], mode="lines+markers",
                               line=dict(color="#111111", width=3, dash="dot"), marker=dict(size=9),
                               name="Pedidos afectados (%)"), secondary_y=True)
    fig_d.add_hline(y=50, line_dash="dash", line_color="#D62728", secondary_y=True,
                    annotation_text="50% de los pedidos", annotation_position="top left")
    fig_d.update_layout(title_text="Beneficio vs. pedidos afectados por escenario de tope", height=470,
                        legend=dict(orientation="h", y=-0.25))
    fig_d.update_yaxes(title_text="Beneficio de los 8 productos ($)", secondary_y=False)
    fig_d.update_yaxes(title_text="Pedidos afectados (%)", range=[0, 105], secondary_y=True)
    st.plotly_chart(estilo_grafico(fig_d), use_container_width=True)

    # ¿Qué tan sensible tendría que ser la demanda para que el tope de 10% deje de convenir?
    lo, hi = -30.0, 0.0
    if beneficio_con_tope(0.10, lo)["beneficio"] < beneficio_actual:
        for _ in range(60):
            mid = (lo + hi) / 2
            if beneficio_con_tope(0.10, mid)["beneficio"] > beneficio_actual:
                hi = mid
            else:
                lo = mid
        e_equilibrio = (lo + hi) / 2
        texto_eq = f"{e_equilibrio:.1f}"
    else:
        texto_eq = "menor a −30"

    # ¿Por qué el tope de 10% es el escenario ideal?
    b_por_banda = bandas.set_index("Banda de descuento")["Beneficio promedio ($)"]
    f20 = tabla_desc[tabla_desc["Escenario"] == "Tope 20%"].iloc[0]
    f15 = tabla_desc[tabla_desc["Escenario"] == "Tope 15%"].iloc[0]
    f5 = tabla_desc[tabla_desc["Escenario"] == "Tope 5%"].iloc[0]
    f0 = tabla_desc[tabla_desc["Escenario"] == "Sin descuentos"].iloc[0]
    st.markdown(f"""
**¿Por qué el tope de 10% es el escenario ideal?**

- **Topes más altos (15% y 20%) ganan menos:** {f15["Variación del beneficio (%)"]:+.1f}% y {f20["Variación del beneficio (%)"]:+.1f}% de beneficio,
  porque dejan vigentes descuentos que destruyen margen.
- **Topes más bajos (5% y sin descuentos) ganan más, pero afectan a la mayoría de clientes:** el {f5["Pedidos afectados (%)"]:.0f}% y el
  {f0["Pedidos afectados (%)"]:.0f}% de los pedidos perderían parte o todo su descuento. Eso eleva el riesgo de perder clientes o
  competitividad, algo que los datos históricos no permiten medir.
- **El tope de 10% es el equilibrio:** logra {fila_ideal["Variación del beneficio (%)"]:+.1f}% de beneficio afectando a **menos de la mitad**
  de los pedidos ({fila_ideal["Pedidos afectados (%)"]:.1f}%). Además, hasta 10% de descuento el beneficio por línea casi no cambia
  (\\${b_por_banda.get("0–5%", float("nan")):.2f} con 0–5% y \\${b_por_banda.get("6–10%", float("nan")):.2f} con 6–10%), mientras que con
  descuentos mayores cae a \\${b_por_banda.get("11–15%", float("nan")):.2f} y \\${b_por_banda.get("16–25%", float("nan")):.2f}: el tope elimina justo
  los descuentos que destruyen margen.
- **Implementación gradual:** si una prueba piloto confirma que la cantidad vendida no cae, más adelante se puede evaluar el tope de 5%.
""")

    st.success(f"⭐ Escenario ideal: limitar los descuentos al 10% incrementaría el beneficio de los productos principales en "
               f"{fila_ideal['Variación del beneficio (%)']:.1f}% (${fila_ideal['Beneficio ($)'] - beneficio_actual:,.0f}), afectando al "
               f"{fila_ideal['Pedidos afectados (%)']:.1f}% de los pedidos y con un cambio en la cantidad vendida de "
               f"{fila_ideal['Cambio en cantidad (%)']:+.2f}%.")
    st.info(f"**Sensibilidad:** el tope de 10% solo dejaría de ser rentable si la elasticidad precio fuera de **{texto_eq}** o "
            f"más negativa (demanda extremadamente sensible). Las elasticidades estimadas están entre "
            f"{e_por_producto.min():.3f} y {e_por_producto.max():.3f}, muy lejos de ese valor.")
    st.caption("Supuesto: la cantidad responde al precio según la elasticidad estimada con datos históricos. "
               "Se recomienda validarlo con una prueba piloto antes de aplicarlo.")


# ==========================================================
# PIE
# ==========================================================
st.markdown("---")
st.caption("Proyecto de Modelamiento Predictivo · DataCo Smart Supply Chain")